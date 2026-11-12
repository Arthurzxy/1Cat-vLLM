# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure fixed-work cache reuse, concurrent capacity and mixed workloads."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--lengths", type=int, nargs="+", default=[131072, 250000])
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / "bench_results/v152_opt_20261111"
source = out / "unique-corpus-tokens.json"
corpus = json.loads(source.read_text())["tokens"]
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
fixtures = out / "cache-performance-inputs"
fixtures.mkdir(exist_ok=True)
report = {"complete": False, "source_sha256": source_hash, "runs": []}
if args.output.exists():
    raise FileExistsError(args.output)
data_dir = args.output.with_suffix(".cases")
data_dir.mkdir(exist_ok=False)
args.output.write_text(json.dumps(report, indent=2) + "\n")


def run_benchmark(label, lengths, concurrency, repeats, tokens_file=None, keep=False):
    result = data_dir / (label + ".json")
    command = [
        str(root / ".host-venv/bin/python"),
        "benchmarks/benchmark_v152.py",
        "--output",
        str(result),
        "--lengths",
        *map(str, lengths),
        "--repeats",
        str(repeats),
        "--output-tokens",
        "1024",
        "--concurrency",
        str(concurrency),
    ]
    if tokens_file:
        command.extend(["--tokens-file", str(tokens_file)])
    if keep:
        command.append("--keep-prefix-cache")
    with result.with_suffix(".log").open("w") as log:
        subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    value = json.loads(result.read_text())
    assert value["complete"]
    assert all(
        len(item["tokens"]) == 1024 for run in value["runs"] for item in run["results"]
    )
    return value


def run_case(label, tokens, keep):
    fixture = fixtures / (label + ".json")
    payload = {"tokens": tokens, "parent_fixture_sha256": source_hash}
    fixture.write_text(json.dumps(payload) + "\n")
    value = run_benchmark(label, [len(tokens)], 1, 1, fixture, keep)
    row = value["runs"][0]
    return {
        "label": label,
        "result_file": str(data_dir / (label + ".json")),
        "run": row,
    }


def rpc(method, kwargs=None):
    request = urllib.request.Request(
        "http://127.0.0.1:18111/collective_rpc",
        data=json.dumps(
            {"method": method, "kwargs": kwargs or {}, "timeout": 120}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


for concurrency in [1, 2, 3, 8]:
    run_benchmark("warm-c" + str(concurrency), [1024], concurrency, 1)
report["runtime"] = rpc("v152_runtime_configuration")
assert all(row["rejection_graphs"] == 3 for row in report["runtime"]["results"])
report["memory_before_prefix_cases"] = rpc("longctx_memory", {"reset": True})
args.output.write_text(json.dumps(report, indent=2) + "\n")


for length in args.lengths:
    base = corpus[: length - 64] + corpus[-64:]
    append = base[:-64] + corpus[-256:-128] + base[-64:]
    cut = length - length // 10
    replacement = length - 64 - cut
    rewritten = base[:cut] + corpus[-replacement - 64 : -64] + base[-64:]
    assert len(base) == len(rewritten) == length
    assert len(append) == length + 128
    for trial in range(args.repeats):
        rows = []
        for kind, tokens, keep in [
            ("cold", base, False),
            ("repeat", base, True),
            ("append", append, True),
            ("rewrite", rewritten, True),
        ]:
            label = f"{length}-{trial}-{kind}"
            rows.append(run_case(label, tokens, keep))
        report["runs"].append(
            {
                "source_length": length,
                "trial": trial,
                "cases": rows,
                "repeat_tokens_equal": rows[0]["run"]["results"][0]["tokens"]
                == rows[1]["run"]["results"][0]["tokens"],
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(length, trial, "cache sequence completed", flush=True)

report["memory_after_prefix_cases"] = rpc("longctx_memory")
rpc("longctx_memory", {"reset": True})
args.output.write_text(json.dumps(report, indent=2) + "\n")
report["concurrent_capacity"] = run_benchmark("c8-capacity", [1024, 32768], 8, 3)
report["memory_after_capacity"] = rpc("longctx_memory")
rpc("longctx_memory", {"reset": True})
args.output.write_text(json.dumps(report, indent=2) + "\n")
mixed_path = data_dir / "mixed-load.json"
with mixed_path.with_suffix(".log").open("w") as log:
    subprocess.run(
        [
            str(root / ".host-venv/bin/python"),
            "benchmarks/mixed_v152.py",
            "--output",
            str(mixed_path),
        ],
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
report["mixed_load_file"] = str(mixed_path)
report["memory_after_mixed"] = rpc("longctx_memory")
assert json.loads(mixed_path.read_text())["complete"]
report["complete"] = True
args.output.write_text(json.dumps(report, indent=2) + "\n")
