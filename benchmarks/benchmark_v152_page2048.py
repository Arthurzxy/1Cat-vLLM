# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare page2048 addressing against generic addressing on identical KV."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmarks.kernels.benchmark_sm70_grouped_attention_long import (  # noqa: E402
    byte_equal,
    call,
    load_operator,
    make_case,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--correctness-only", action="store_true")
parser.add_argument("--sanitizer-smoke", action="store_true")
parser.add_argument("--native-library", type=Path)
parser.add_argument("--reference-manifest", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--run-dir", type=Path, default=root / "bench_results/page2048-r1")
args = parser.parse_args()
directory = args.run_dir.resolve()
manifest = json.loads((directory / "manifest.json").read_text())
library = Path(manifest["library"])
assert hashlib.sha256(library.read_bytes()).hexdigest() == manifest["library_sha256"]
spec = importlib.util.spec_from_file_location(manifest["module_name"], library)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
operators = [module.baseline, module.run]
if args.reference_manifest:
    operators[0], reference = load_operator(args.reference_manifest)
    manifest = {**manifest, "reference": reference}
if args.native_library:
    torch.ops.load_library(str(args.native_library))
    operators[1] = torch.ops._vllm_fa2_C.sm70_grouped_long_fwd
    manifest = {
        **manifest,
        "native_library": str(args.native_library),
        "native_sha256": hashlib.sha256(args.native_library.read_bytes()).hexdigest(),
    }
report = {
    "manifest": manifest,
    "checks": [],
    "performance": [],
    "scope": "operator only; full-model admission pending",
    "complete": False,
}
output = args.output or directory / (
    "correctness.json" if args.correctness_only else "operator.json"
)
if output.exists():
    raise FileExistsError(output)
assert torch.cuda.get_device_capability() == (7, 0)
torch.manual_seed(20261109)
if hasattr(module, "decode_e4m3"):
    raw_codes = torch.arange(256, device="cuda", dtype=torch.int16).to(torch.uint8)
    actual_decode = module.decode_e4m3(raw_codes)
    expected_decode = raw_codes.view(torch.float8_e4m3fn).half()
    finite_codes = torch.isfinite(expected_decode)
    assert byte_equal(actual_decode[finite_codes], expected_decode[finite_codes])
    assert torch.equal(torch.isnan(actual_decode), torch.isnan(expected_decode))
    report["codec_check"] = {"finite_codes_bit_exact": 254, "nan_codes_preserved": 2}


def save():
    output.write_text(json.dumps(report, indent=2) + chr(10))


pages = (2048,) if args.sanitizer_smoke else (2048, 3296)
query_rows = (8,) if args.sanitizer_smoke else range(2, 9)
lengths = (
    (128, 2049, 8192, 131071)
    if args.sanitizer_smoke
    else (8, 2047, 2048, 2049, 8192, 131071, 262144)
)
for page in pages:
    for rows in query_rows:
        for length in lengths:
            for padding in (0, 8):
                case = make_case(rows, page, length, 2, stride_padding=padding)
                graphs = []
                for arm, operator in enumerate(operators):
                    call(case, arm, operator)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        call(case, arm, operator)
                    graphs.append(graph)
                for state in ("original", "zero", "restored"):
                    if state == "zero":
                        case["lengths"].zero_()
                    else:
                        case["lengths"].copy_(case["initial_lengths"])
                    for graph in graphs:
                        graph.replay()
                    torch.accelerator.synchronize()
                    equal = all(
                        byte_equal(case["arms"][0][key], case["arms"][1][key])
                        for key in ("out", "partial", "lse")
                    )
                    finite = all(
                        bool(torch.isfinite(a["out"]).all()) for a in case["arms"]
                    )
                    guards = all(
                        bool((a["guard"][[0, -1]] == -777).all()) for a in case["arms"]
                    )
                    row = dict(
                        page=page,
                        rows=rows,
                        length=length,
                        padding=padding,
                        state=state,
                        byte_equal=equal,
                        finite=finite,
                        guards=guards,
                    )
                    report["checks"].append(row)
                    if not (equal and finite and guards):
                        save()
                        raise AssertionError(row)
                del graphs, case
save()
print("correctness checks", len(report["checks"]), flush=True)
if args.correctness_only:
    report["complete"] = True
    save()
    raise SystemExit(0)
for length in (8192, 32768, 65536, 131072, 250000):
    cases = [make_case(8, 2048, length, 2) for _ in range(16)]
    graphs = []
    for arm, operator in enumerate(operators):
        for case in cases:
            call(case, arm, operator)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for case in cases:
                call(case, arm, operator)
        graphs.append(graph)
    for _ in range(10):
        for graph in graphs:
            graph.replay()
    torch.accelerator.synchronize()
    samples = [[], []]
    for _ in range(5):
        for arm in (0, 1, 1, 0):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(3):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(begin.elapsed_time(end) / 3)
    medians = [statistics.median(x) for x in samples]
    row = dict(
        length=length,
        layers=16,
        samples_ms=samples,
        median_ms=medians,
        latency_reduction=1 - medians[1] / medians[0],
    )
    report["performance"].append(row)
    save()
    print(json.dumps(row), flush=True)
    del cases, graphs
report["complete"] = True
save()
