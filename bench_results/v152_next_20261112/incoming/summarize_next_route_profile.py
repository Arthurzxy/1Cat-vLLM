# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank GPU kernels linked to exact fixed-policy CPU external IDs."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
out = args.root
folder = out / "service-canary/profiles"
report = {
    "complete": False,
    "ranks": [],
    "scope": (
        "One complete cold131072-input/64-output canary with final maxseq8 "
        "geometry. Timings are profiler diagnostics. Four ranks are reported "
        "separately."
    ),
    "window_definition": (
        "Target fixed-policy GEMMs alone, indexed by128 target calls per8192-token "
        "chunk. Early chunks0-3, middle4-11, late12-15. No claim that a target-"
        "kernel sum is a complete request critical path."
    ),
    "source_owner": (
        "Qpn2NvFp4LinearKernel.apply_qpn2 -> installed "
        "_C.nvfp4_qpn2_tm_dispatch_sm70_out -> sm70_nvfp4_prefill_lt -> cuBLASLt "
        "fixed row-major K,N plan."
    ),
    "layout_contract": (
        "FP16 contiguous[M,5120] times FP16 contiguous[5120,N], FP32 accumulation, "
        "zero workspace. Captured checkpoint weight/scale ownership is in "
        "REAL_PROJECTION_INPUTS.json."
    ),
}
for path in sorted(folder.glob("*rank*.json")):
    events = json.loads(path.read_bytes())["traceEvents"]
    owners = sorted(
        [
            event
            for event in events
            if event.get("ph") == "X" and event.get("name") == "sm70_nvfp4_prefill_lt"
        ],
        key=lambda event: event["ts"],
    )
    assert len(owners) == 2048
    by_external = {
        event["args"]["External id"]: (index, event)
        for index, event in enumerate(owners)
    }
    grouped = collections.defaultdict(list)
    all_kernels = [
        event
        for event in events
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    for kernel in all_kernels:
        external = kernel.get("args", {}).get("External id")
        if external not in by_external:
            continue
        index, owner = by_external[external]
        dims = owner["args"]["Input Dims"]
        assert dims[0] == [8192, 5120] and dims[1][0] == 5120
        n = dims[1][1]
        chunk = index // 128
        stage = "early" if chunk < 4 else "middle" if chunk < 12 else "late"
        grouped[(n, stage)].append(kernel)
    projection = {
        3584: "full_attention_qkv",
        4128: "gdn_qkvzba_padded4128",
        8704: "mlp_gate_up",
    }
    summaries = []
    for (n, stage), kernels in sorted(grouped.items()):
        summaries.append(
            {
                "projection": projection[n],
                "m": 8192,
                "k": 5120,
                "n": n,
                "stage": stage,
                "kernel_count": len(kernels),
                "gpu_kernel_sum_ms": sum(kernel["dur"] for kernel in kernels) / 1000,
                "symbols": dict(
                    collections.Counter(kernel["name"] for kernel in kernels)
                ),
                "registers_per_thread": sorted(
                    {kernel["args"]["registers per thread"] for kernel in kernels}
                ),
                "shared_memory_bytes": sorted(
                    {kernel["args"]["shared memory"] for kernel in kernels}
                ),
                "blocks": sorted(
                    {tuple(kernel["args"]["block"]) for kernel in kernels}
                ),
            }
        )
    matched = sum(row["kernel_count"] for row in summaries)
    assert matched == 2048, matched
    rank = int(path.stem.rsplit("rank", 1)[1])
    report["ranks"].append(
        {
            "rank": rank,
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fixed_policy_cpu_calls": len(owners),
            "exact_external_id_gpu_matches": matched,
            "all_profile_kernel_count": len(all_kernels),
            "all_profile_gpu_kernel_sum_ms": sum(
                kernel["dur"] for kernel in all_kernels
            )
            / 1000,
            "fixed_policy_kernel_sum_ms": sum(
                row["gpu_kernel_sum_ms"] for row in summaries
            ),
            "selected_shapes": summaries,
        }
    )
    print("rank", rank, "matched", matched, flush=True)
    del events, owners, by_external, all_kernels, grouped
report["complete"] = True
(out / "FINAL_ROUTE_PROFILE.json").write_text(json.dumps(report, indent=2) + "\n")
