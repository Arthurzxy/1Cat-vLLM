# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute cache, mixed-load and scheduler-age results from raw files."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--source-root", type=Path)
args = parser.parse_args()
out = args.root.resolve()
source_root = args.source_root
if source_root is None:
    source_root = out.parent / "source-final"
    if not source_root.exists():
        source_root = out.parents[1]
sys.path.insert(0, str(source_root / "benchmarks"))
admission = json.loads((out / "EFFICIENCY_SERVICE_RUNS_COMPLETE.json").read_text())
assert admission["complete"]


def quantile(values, fraction):
    values = sorted(values)
    assert values
    return values[math.ceil(fraction * (len(values) - 1))]


def metric_delta(before, after):
    names = {
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:request_prefill_kv_computed_tokens_sum",
        "vllm:num_preemptions_total",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_queue_time_seconds_sum",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    }
    snapshots = []
    for body in [before, after]:
        values = {}
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            name, value = line.rsplit(" ", 1)
            key = name.split("{", 1)[0]
            if key not in names:
                continue
            position = name.partition('position="')[2].partition('"')[0]
            if position:
                key += "/position=" + position
            values[key] = values.get(key, 0) + float(value)
        snapshots.append(values)
    return {
        key: snapshots[1].get(key, 0) - snapshots[0].get(key, 0)
        for key in snapshots[0].keys() | snapshots[1].keys()
    }


def read_lines(paths):
    return [
        json.loads(line) for path in paths for line in path.read_text().splitlines()
    ]


def pool_summary(records):
    partitions = {}
    for record in records:
        for name, values in record["partitions"].items():
            target = partitions.setdefault(
                name,
                {
                    "capacity_blocks": values["capacity_blocks"],
                    "pinned_blocks_max": 0,
                    "free_or_evictable_blocks_min": values["capacity_blocks"],
                    "cached_physical_blocks_max": 0,
                },
            )
            assert target["capacity_blocks"] == values["capacity_blocks"]
            assert 0 <= values["pinned_blocks"] <= values["capacity_blocks"]
            target["pinned_blocks_max"] = max(
                target["pinned_blocks_max"], values["pinned_blocks"]
            )
            target["free_or_evictable_blocks_min"] = min(
                target["free_or_evictable_blocks_min"],
                values["free_or_evictable_blocks"],
            )
            target["cached_physical_blocks_max"] = max(
                target["cached_physical_blocks_max"],
                values.get("cached_physical_blocks", 0),
            )
    return {
        "records": len(records),
        "partitions": partitions,
        "partial_owned_blocks_max": max(
            (row["partial_owned_blocks"] for row in records), default=0
        ),
        "partial_entries_max": max(
            (row["partial_entries"] for row in records), default=0
        ),
        "partial_block_limits": sorted({row["partial_block_limit"] for row in records}),
        "count_scope": (
            "Cached physical blocks can be free/evictable or pinned. Active group "
            "references can share IDs. Counts retain those distinct meanings."
        ),
    }


def fixed_cases(folder, pool_records, clock_offset):
    path = folder / "fixed.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text())
    assert value["complete"]
    rows = []
    by_sequence = {}
    for case in value["runs"]:
        source = folder / "fixed.cases" / Path(case["file"]).name
        run = json.loads(source.read_text())["runs"][0]
        result = run["results"][0]
        assert len(result["tokens"]) == run["emitted_tokens"] == 1024
        begin = result["started_monotonic_s"] + clock_offset
        end = result["finished_monotonic_s"] + clock_offset
        lookups = [
            record
            for record in pool_records
            if record["stage"] == "lookup"
            and begin - 0.05 <= record["time"] <= end + 0.05
        ]
        request_ids = {record["request_id"] for record in lookups}
        associated = [
            record for record in pool_records if record["request_id"] in request_ids
        ]
        metric = run["metrics"]
        rows.append(
            {
                "source_length": case["length"],
                "trial": case["trial"],
                "kind": case["kind"],
                "input_tokens": result["prompt_tokens"],
                "actual_output_tokens": len(result["tokens"]),
                "computed_input_tokens": metric.get(
                    "vllm:request_prefill_kv_computed_tokens_sum"
                ),
                "prefix_cache_hits_total": metric.get("vllm:prefix_cache_hits_total"),
                "preemptions": metric.get("vllm:num_preemptions_total"),
                "client_ttft_s": result["ttft_s"],
                "complete_request_output_tps": run["aggregate_output_tps"],
                "server_prefill_s": metric.get("vllm:request_prefill_time_seconds_sum"),
                "server_decode_s": metric.get("vllm:request_decode_time_seconds_sum"),
                "pool_request_ids": sorted(request_ids),
                "pool_observation": pool_summary(associated) if associated else None,
                "rank_memory_before": case["memory_before"],
                "rank_memory_after": case["memory_after"],
            }
        )
        by_sequence.setdefault((case["length"], case["trial"]), {})[case["kind"]] = (
            result["tokens"]
        )
    equality = [
        {
            "source_length": key[0],
            "trial": key[1],
            "cold_repeat_tokens_equal": sequence["cold"] == sequence["repeat"],
        }
        for key, sequence in by_sequence.items()
    ]
    return {"rows": rows, "cold_repeat_comparisons": equality}


def chat_cases(path, pool_records):
    value = json.loads(path.read_text())
    assert value["complete"]
    rows = []
    for check in value["checks"]:
        response = check["response"]
        request_id = response["id"]
        associated = [
            record
            for record in pool_records
            if record["request_id"] == request_id
            or record["request_id"].startswith(request_id + "-")
        ]
        metrics = metric_delta(check["metrics_before"], check["metrics_after"])
        rows.append(
            {
                "label": check["label"],
                "passed": check["passed"],
                "input_tokens": response["usage"]["prompt_tokens"],
                "actual_output_tokens": response["usage"]["completion_tokens"],
                "server_metric_ttft_s": metrics.get(
                    "vllm:time_to_first_token_seconds_sum"
                ),
                "wall_s": check["wall_s"],
                "metrics": metrics,
                "pool_observation": pool_summary(associated) if associated else None,
            }
        )
    return {"all_reference_checks_passed": value["passed"], "rows": rows}


cache = {}
for folder in sorted((out / "efficiency").iterdir()):
    if not (folder / "complete.json").exists():
        continue
    if not (folder / "fixed.json").exists() and not list(folder.glob("chat*.json")):
        continue
    identity = json.loads((folder / "identity.json").read_text())
    pool_records = read_lines(sorted((folder / "cache-trace").glob("pools-*.jsonl")))
    clock = identity["configuration"]["results"][0]["clock"]
    clock_offset = clock["epoch_s"] - clock["monotonic_s"]
    cache[folder.name] = {
        "image_id": identity["image"],
        "instrumented": identity["instrumented"],
        "configuration": identity["configuration"],
        "pool_observation": pool_summary(pool_records),
        "fixed": fixed_cases(folder, pool_records, clock_offset),
        "chat": {
            path.stem: chat_cases(path, pool_records)
            for path in sorted(folder.glob("chat*.json"))
            if ".stage." not in path.name
        },
        "independent_service_processes": 1,
    }


def mixed_process(folder):
    from v152_validation import chunk_intervals_after

    value = json.loads((folder / "mixed.json").read_text())
    assert value["complete"]
    trials = []
    all_short = []
    all_gaps = []
    for index, trial in enumerate(value["trials"]):
        requests = trial["requests"]
        resident = next(row for row in requests if row["label"] == "resident-long")
        short = [row for row in requests if row["label"].startswith("short-")]
        medium = next(row for row in requests if row["label"] == "insert-32k")
        assert len(short) == 12
        assert (
            resident["start"] + resident["ttft_s"]
            <= trial["inserted"]
            < resident["end"]
        )
        gaps = chunk_intervals_after(
            resident["chunks"], resident["start"], trial["inserted"]
        )
        outputs = sum(len(row["tokens"]) for row in requests)
        assert outputs == 4096
        whole = max(row["end"] for row in requests) - min(
            row["start"] for row in requests
        )
        all_short.extend(row["ttft_s"] for row in short)
        all_gaps.extend(gaps)
        trials.append(
            {
                "trial": index,
                "short_ttft_s": [row["ttft_s"] for row in short],
                "short_empirical_p95_s": quantile(
                    [row["ttft_s"] for row in short], 0.95
                ),
                "resident_initial_ttft_s": resident["ttft_s"],
                "resident_complete_wall_s": resident["end"] - resident["start"],
                "resident_chunk_gap_p95_s": quantile(gaps, 0.95),
                "resident_chunk_gap_max_s": max(gaps),
                "insert32k_ttft_s": medium["ttft_s"],
                "insert32k_complete_wall_s": medium["end"] - medium["start"],
                "actual_output_tokens": outputs,
                "whole_workload_wall_s": whole,
                "whole_workload_output_tps": outputs / whole,
                "metrics": metric_delta(
                    trial["metrics_before"], trial["metrics_after"]
                ),
            }
        )
    return {
        "independent_service_processes": 1,
        "trials": trials,
        "short_count": len(all_short),
        "short_empirical_p95_s": quantile(all_short, 0.95),
        "resident_chunk_gap_p95_s": quantile(all_gaps, 0.95),
        "resident_chunk_gap_max_s": max(all_gaps),
        "mean_whole_workload_output_tps": statistics.mean(
            row["whole_workload_output_tps"] for row in trials
        ),
        "mean_resident_complete_wall_s": statistics.mean(
            row["resident_complete_wall_s"] for row in trials
        ),
        "mean_insert32k_ttft_s": statistics.mean(
            row["insert32k_ttft_s"] for row in trials
        ),
    }


mixed = {
    folder.name: mixed_process(folder)
    for folder in sorted((out / "efficiency").iterdir())
    if (folder / "mixed.json").exists() and (folder / "complete.json").exists()
}
trace = read_lines(
    sorted((out / "efficiency/async-age-diagnostic/scheduler-trace").glob("*.jsonl"))
)
assert trace, "Actual scheduler-age observation is required"
schedules = [row for row in trace if row["event"] == "schedule"]
feedback = [
    row["feedback"] for row in trace if row["event"] == "update" and row["feedback"]
]
assert schedules and feedback
ages = [
    request["last_output_age_s"]
    for row in schedules
    for request in row["running_before"]
    if request["output_tokens"] > 0 and request["last_output_age_s"] is not None
]
assert ages
resident_ids = {
    request["request_id"]
    for row in schedules
    for request in row["running_before"]
    if request["prompt_tokens"] == 131072
}
assert len(resident_ids) == 1
diagnostic = json.loads(
    (out / "efficiency/async-age-diagnostic/mixed.json").read_text()
)
assert len(diagnostic["trials"]) == 1
injected = diagnostic["trials"][0]["inserted"]
resident_updates = [
    {"at": row["at"], "tokens": request["committed_tokens"]}
    for row in trace
    if row["event"] == "update"
    for request in row["emitted"]
    if request["request_id"] in resident_ids
]
resident_updates.sort(key=lambda row: row["at"])
expected_tokens = len(diagnostic["trials"][0]["requests"][0]["tokens"])
assert sum(row["tokens"] for row in resident_updates) == expected_tokens == 2048
commit_intervals = [
    {"end": right["at"], "elapsed_s": right["at"] - left["at"]}
    for left, right in zip(resident_updates, resident_updates[1:])
]
commit_gaps = [row["elapsed_s"] for row in commit_intervals]
injected_gaps = [row["elapsed_s"] for row in commit_intervals if row["end"] >= injected]
assert commit_gaps and injected_gaps
age_summary = {
    "scheduler_classes": sorted({row["class"] for row in schedules}),
    "schedule_records": len(schedules),
    "completed_feedback_records": len(feedback),
    "post_commit_schedule_age_p95_s": quantile(ages, 0.95),
    "post_commit_schedule_age_max_s": max(ages),
    "resident_request_ids": sorted(resident_ids),
    "resident_cpu_committed_tokens": expected_tokens,
    "resident_cpu_commit_gap_p95_s": quantile(commit_gaps, 0.95),
    "resident_cpu_commit_gap_max_s": max(commit_gaps),
    "resident_cpu_commit_gap_after_injection_p95_s": quantile(injected_gaps, 0.95),
    "resident_cpu_commit_gap_after_injection_max_s": max(injected_gaps),
    "mixed_schedule_records": sum(
        row["mixed_prefill_tokens"] > 0 and row["mixed_decode_tokens"] > 0
        for row in schedules
    ),
    "observed_controller_budget_tokens": sorted(
        {row["controller_tokens"] for row in schedules}
    ),
    "feedback_elapsed_ms_min": min(row["elapsed_ms"] for row in feedback),
    "feedback_elapsed_ms_max": max(row["elapsed_ms"] for row in feedback),
    "scope": (
        "The schedule-entry age is sampled immediately after output processing "
        "and describes CPU loop delay. Actual resident progress stalls are "
        "recomputed between successive nonempty CPU commit callbacks, including "
        "the interval crossing injection. These intervals include GPU waiting "
        "and CPU processing and retain their diagnostic scope. Nonblocking GPU "
        "feedback is recorded separately."
    ),
}
selection = json.loads((out / "ASYNC_SELECTION.json").read_text())
report = {
    "complete": True,
    "cache": cache,
    "mixed": mixed,
    "scheduler_age_diagnostic": age_summary,
    "async_selection": selection,
    "sampling_scope": (
        "Fresh service process is the independent unit. Three mixed trials yield36 "
        "short requests/process. Empirical p95 uses ceil(.95*(n-1)) in sorted "
        "observations. Formal mixed/cache timing and diagnostic pool/age passes "
        "retain separate identities."
    ),
    "kernel_combination_image": admission["selected_kernel_image"],
}
(out / "EFFICIENCY_SUMMARY.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
)
print(
    json.dumps(
        {
            "complete": True,
            "cache_profiles": list(cache),
            "mixed_processes": {
                name: {
                    "p95_s": row["short_empirical_p95_s"],
                    "resident_max_gap_s": row["resident_chunk_gap_max_s"],
                    "whole_output_tps": row["mean_whole_workload_output_tps"],
                }
                for name, row in mixed.items()
            },
            "retain_optional_async_preset": selection["retain_optional_preset"],
        }
    ),
    flush=True,
)
