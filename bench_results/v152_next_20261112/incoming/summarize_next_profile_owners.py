# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One attribution per GPU kernel, using CPU external IDs and containment."""

import argparse
import collections
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
out = args.root
report = {
    "complete": False,
    "ranks": [],
    "scope": (
        "Complete cold131072-input/64-output canary. Each GPU kernel contributes "
        "once to its exact CPU external-ID bucket. Parent native names provide "
        "context only. Per-rank kernel sums describe structure within this profiled"
        " request."
    ),
    "timing_units": "milliseconds",
    "phase_scope": (
        "The canary includes all16 prefill chunks and its64-token completion. "
        "Dedicated strict-decode boundaries remain in the earlier slice and batch-"
        "ledger diagnostics."
    ),
}
for path in sorted((out / "service-canary/profiles").glob("*rank*.json")):
    events = json.loads(path.read_bytes())["traceEvents"]
    threads = collections.defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in ["cpu_op", "user_annotation"]:
            threads[(event["pid"], event["tid"])].append(event)
    owners = {}
    for thread in threads.values():
        stack = []
        for event in sorted(thread, key=lambda item: (item["ts"], -item["dur"])):
            end = event["ts"] + event["dur"]
            while stack and not (
                stack[-1]["ts"] <= event["ts"]
                and end <= stack[-1]["ts"] + stack[-1]["dur"] + 0.001
            ):
                stack.pop()
            external = event.get("args", {}).get("External id")
            if external is not None:
                native = next(
                    (
                        parent["name"]
                        for parent in reversed(stack)
                        if parent["name"].startswith(("_C::", "vllm::"))
                    ),
                    None,
                )
                dims = event.get("args", {}).get("Input Dims")
                owners[external] = (event["name"], native, json.dumps(dims))
            stack.append(event)
    buckets = {}
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    for event in kernels:
        external = event.get("args", {}).get("External id")
        owner = owners.get(external, ("graph_or_external_id_unavailable", None, "null"))
        if owner not in buckets:
            buckets[owner] = {"count": 0, "us": 0, "symbols": collections.Counter()}
        row = buckets[owner]
        row["count"] += 1
        row["us"] += event["dur"]
        row["symbols"][event["name"]] += 1
    total = sum(event["dur"] for event in kernels)
    assert sum(row["count"] for row in buckets.values()) == len(kernels)
    rows = []
    for (name, native, dims), values in sorted(
        buckets.items(), key=lambda item: item[1]["us"], reverse=True
    ):
        rows.append(
            {
                "cpu_leaf": name,
                "native_parent_context": native,
                "input_dims": json.loads(dims),
                "gpu_kernel_count": values["count"],
                "gpu_kernel_sum_ms": values["us"] / 1000,
                "share_of_same_rank_profile_kernel_sum_pct": 100 * values["us"] / total,
                "symbols": dict(values["symbols"]),
            }
        )
    rank = int(path.stem.rsplit("rank", 1)[1])
    report["ranks"].append(
        {
            "rank": rank,
            "gpu_kernel_count": len(kernels),
            "gpu_kernel_sum_ms": total / 1000,
            "attribution": rows,
        }
    )
    print(
        rank,
        [
            (
                row["cpu_leaf"],
                row["native_parent_context"],
                round(row["share_of_same_rank_profile_kernel_sum_pct"], 3),
            )
            for row in rows[:8]
        ],
        flush=True,
    )
    del events, threads, owners, buckets, kernels
report["complete"] = True
(out / "FINAL_HOTSPOT_OWNERS.json").write_text(json.dumps(report, indent=2) + "\n")
