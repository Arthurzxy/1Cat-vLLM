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
args = parser.parse_args()
out = args.root
groups = {
    "prev": ["prev-p1", "prev-p2"],
    "safe": ["safe-a1", "safe-a2"],
    "new": ["new-b1", "new-b2"],
}
cells = {}
raw = {}
sources = {}
for arm, processes in groups.items():
    for process in processes:
        folder = out / "formal" / process
        if not (folder / "complete.json").exists():
            continue
        for source in sorted(folder.glob("formal-c*.json")):
            if ".stage." in source.name:
                continue
            value = json.loads(source.read_text())
            if not value["complete"]:
                continue
            sources[str(source.relative_to(out))] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
            concurrency = int(source.stem.removeprefix("formal-c"))
            for row in value["runs"]:
                key = f"C{concurrency}/{row['length']}"
                metrics = row["metrics"]

                def metric(name, metrics=metrics):
                    return metrics.get("vllm:" + name, 0)

                prefill = metric("request_prefill_time_seconds_sum") / concurrency
                decode = metric("request_decode_time_seconds_sum") / concurrency
                computed = metric("request_prefill_kv_computed_tokens_sum")
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
                raw[(arm, key, row["trial"])] = [x["tokens"] for x in row["results"]]


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
        for name in metric_names[:6]:
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
complete = (out / "FORMAL_SERVICE_ADMISSION.json").exists()
report = {
    "complete": complete,
    "cells": summary,
    "raw_sources_sha256": sources,
    "independent_unit": (
        "Fresh service process. Two batches per cell are nested measurements."
    ),
    "ordering": "Bprev P1, Bsafe A1, Bnew B1, Bnew B2, Bsafe A2, Bprev P2",
    "limits": [
        "Two processes per arm do not establish a statistical no-regression guarantee.",
        (
            "Request decode/prefill durations overlap for concurrent requests. "
            "Their sums are not system throughput denominators."
        ),
        (
            "Computed prefill TPS is a per-request effective rate including "
            "scheduling, not pure CUDA kernel rate."
        ),
        (
            "Actual GPU step/bonus/truncation accounting is in the separate "
            "instrumented batch ledger."
        ),
    ],
}
(out / "FORMAL_SERVICE_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
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
