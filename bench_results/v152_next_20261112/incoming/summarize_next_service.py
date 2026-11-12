# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute process-level service results from immutable raw benchmark files."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--study", choices=["main", "short"], default="main")
args = parser.parse_args()
out = args.root
groups = {
    "prev": ["prev-p1", "prev-p2"],
    "safe": ["safe-a1", "safe-a2"],
    "new": ["new-b1", "new-b2"],
}
study_folder = "formal"
pattern = "formal-c*.json"
marker = "FORMAL_SERVICE_ADMISSION.json"
summary_name = "FORMAL_SERVICE_SUMMARY.json"
ordering = "Bprev P1, Bsafe A1, Bnew B1, Bnew B2, Bsafe A2, Bprev P2"
if args.study == "short":
    groups = {"safe": ["safe-q1", "safe-q2"], "new": ["new-r1", "new-r2"]}
    study_folder = "short-confirm"
    pattern = "confirm-c*.json"
    marker = "SHORT_CONFIRM_ADMISSION.json"
    summary_name = "SHORT_CONFIRM_SUMMARY.json"
    ordering = "Bsafe Q1, Bnew R1, Bnew R2, Bsafe Q2"
cells = {}
sources = {}
for arm, processes in groups.items():
    for process in processes:
        folder = out / study_folder / process
        if not (folder / "complete.json").exists():
            continue
        for source in sorted(folder.glob(pattern)):
            if ".stage." in source.name:
                continue
            value = json.loads(source.read_text())
            if not value["complete"]:
                continue
            sources[str(source.relative_to(out))] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
            concurrency = int(source.stem.split("-")[1].removeprefix("c"))
            for row in value["runs"]:
                key = f"C{concurrency}/{row['length']}"
                metrics = row["metrics"]

                def metric(name, metrics=metrics):
                    return metrics.get("vllm:" + name, 0)

                prefill = metric("request_prefill_time_seconds_sum") / concurrency
                decode = metric("request_decode_time_seconds_sum") / concurrency
                computed = metric("request_prefill_kv_computed_tokens_sum")
                started = min(item["started_monotonic_s"] for item in row["results"])
                last_first_chunk = (
                    max(
                        item["started_monotonic_s"] + item["ttft_s"]
                        for item in row["results"]
                    )
                    - started
                )
                sample = {
                    "aggregate_output_tps": row["emitted_tokens"] / row["batch_wall_s"],
                    "batch_wall_s": row["batch_wall_s"],
                    "mean_request_ttft_s": statistics.mean(
                        x["ttft_s"] for x in row["results"]
                    ),
                    "mean_request_prefill_s": prefill,
                    "mean_request_decode_s": decode,
                    "mean_request_computed_prefill_tps": computed
                    / concurrency
                    / prefill,
                    "batch_last_ttft_s": last_first_chunk,
                    "computed_input_tokens_per_batch_last_ttft_s": (
                        computed / last_first_chunk
                    ),
                    "mean_request_post_first_chunk_tps": statistics.mean(
                        item["post_first_chunk_tps"] for item in row["results"]
                    ),
                    "computed_input_tokens": computed,
                    "cache_hits": metric("prefix_cache_hits_total"),
                    "preemptions": metric("num_preemptions_total"),
                    "drafts": metric("spec_decode_num_drafts_total"),
                    "proposed": metric("spec_decode_num_draft_tokens_total"),
                    "accepted": metric("spec_decode_num_accepted_tokens_total"),
                    "emitted_tokens": row["emitted_tokens"],
                    "resident_peak": max(x["running"] for x in row["residency"]),
                    "waiting_peak": max(x["waiting"] for x in row["residency"]),
                }
                assert sample["emitted_tokens"] == concurrency * 1024
                assert (
                    computed == concurrency * row["length"]
                    and sample["cache_hits"] == 0
                )
                cells.setdefault(key, {}).setdefault(arm, {}).setdefault(
                    process, []
                ).append(sample)


def describe(values):
    return {
        "process_values": values,
        "mean": statistics.mean(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "independent_processes": len(values),
    }


summary = {}
metric_names = [
    "aggregate_output_tps",
    "batch_wall_s",
    "mean_request_ttft_s",
    "mean_request_prefill_s",
    "mean_request_decode_s",
    "mean_request_computed_prefill_tps",
    "batch_last_ttft_s",
    "computed_input_tokens_per_batch_last_ttft_s",
    "mean_request_post_first_chunk_tps",
    "accepted",
    "proposed",
    "drafts",
    "resident_peak",
    "waiting_peak",
    "preemptions",
]
for key, arms in cells.items():
    record = {"arms": {}, "comparisons": {}}
    for arm, processes in arms.items():
        record["arms"][arm] = {
            name: describe(
                [
                    statistics.median(sample[name] for sample in processes[process])
                    for process in groups[arm]
                    if process in processes
                ]
            )
            for name in metric_names
        }
        record["arms"][arm]["raw_process_batches"] = processes
    for candidate, baseline, label in [
        ("new", "safe", "performance_increment"),
        ("safe", "prev", "correctness_cost"),
    ]:
        if candidate not in arms or baseline not in arms:
            continue
        changes = {}
        for name in metric_names[:9]:
            left = record["arms"][candidate][name]
            right = record["arms"][baseline][name]
            changes[name] = {
                "mean_relative_change_pct": 100 * (left["mean"] / right["mean"] - 1),
                "ordered_process_pair_changes_pct": [
                    100 * (a / b - 1)
                    for a, b in zip(left["process_values"], right["process_values"])
                ],
            }
        record["comparisons"][label] = changes
    summary[key] = record
complete = (out / marker).exists()
report = {
    "complete": complete,
    "cells": summary,
    "raw_sources_sha256": sources,
    "independent_unit": (
        "Fresh service process. Per-process medians summarize nested batches."
    ),
    "ordering": ordering,
    "batch_prefill_timing": (
        "batch_last_ttft_s spans the earliest request start through the last "
        "observed first SSE chunk. Its computed-input rate includes queueing, "
        "transport and concurrent decode work. Per-request server prefill "
        "durations are reported separately."
    ),
    "limits": [
        "Independent service-process count is2/arm. Statistical confidence is limited.",
        (
            "Request decode/prefill durations overlap for concurrent requests. "
            "System throughput uses the complete batch wall interval."
        ),
        (
            "Computed prefill TPS is a per-request effective rate including "
            "scheduling. CUDA kernel timing is available in separate diagnostics."
        ),
        (
            "Actual GPU step/bonus/truncation accounting is in the separate "
            "instrumented batch ledger."
        ),
    ],
}
(out / summary_name).write_text(json.dumps(report, indent=2) + "\n")
for key, cell in summary.items():
    change = cell["comparisons"].get("performance_increment", {})
    print(
        key,
        json.dumps(
            {
                name: round(values["mean_relative_change_pct"], 3)
                for name, values in change.items()
            }
        ),
        flush=True,
    )
print("complete", complete, flush=True)
