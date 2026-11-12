# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single verified q8/131K/S80 attention launch for Nsight Compute counters."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmarks.kernels.benchmark_sm70_grouped_attention_long import (  # noqa: E402
    call,
    make_case,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(20261111)
torch.ops.load_library(str(args.library.resolve()))
case = make_case(8, 2048, 131072, 1, split_counts=[80])
for _ in range(4):
    call(case, 0, torch.ops._vllm_fa2_C.sm70_grouped_long_fwd)
torch.accelerator.synchronize()
print(
    json.dumps(
        {
            "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
            "q": list(case["q"].shape),
            "kv": list(case["k"].shape),
            "splits": 80,
            "finite": bool(torch.isfinite(case["arms"][0]["out"]).all()),
        }
    )
)
