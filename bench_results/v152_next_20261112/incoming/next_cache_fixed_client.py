# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused fixed-output cache sequences, with pool snapshots per request."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--lengths", type=int, nargs="+", default=[131072, 250000])
parser.add_argument("--repeats", type=int, default=2)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
source = root / "bench_results/v152_opt_20261111/unique-corpus-tokens.json"
corpus = json.loads(source.read_text())["tokens"]
assert not args.output.exists()
cases = args.output.with_suffix(".cases")
cases.mkdir()
report = {
    "complete": False,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "runs": [],
}


def memory():
    request = urllib.request.Request(
        "http://127.0.0.1:18111/collective_rpc",
        data=json.dumps({"method": "longctx_memory", "timeout": 120}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


for length in args.lengths:
    base = corpus[: length - 64] + corpus[-64:]
    append = base[:-64] + corpus[-256:-128] + base[-64:]
    cut = length - length // 10
    replacement = length - 64 - cut
    rewrite = base[:cut] + corpus[-replacement - 64 : -64] + base[-64:]
    assert len(base) == len(rewrite) == length and len(append) == length + 128
    for trial in range(args.repeats):
        rows = []
        for kind, tokens in [
            ("cold", base),
            ("repeat", base),
            ("append", append),
            ("rewrite", rewrite),
        ]:
            label = f"{length}-{trial}-{kind}"
            fixture = cases / (label + "-tokens.json")
            fixture.write_text(json.dumps({"tokens": tokens}) + "\n")
            destination = cases / (label + ".json")
            command = [
                str(root / ".host-venv/bin/python"),
                "benchmarks/benchmark_v152.py",
                "--output",
                str(destination),
                "--lengths",
                str(len(tokens)),
                "--repeats",
                "1",
                "--output-tokens",
                "1024",
                "--concurrency",
                "1",
                "--tokens-file",
                str(fixture),
            ]
            if kind != "cold":
                command.append("--keep-prefix-cache")
            before = memory()
            with (cases / (label + ".log")).open("w") as log:
                subprocess.run(
                    command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            value = json.loads(destination.read_text())
            assert value["complete"] and value["runs"][0]["emitted_tokens"] == 1024
            rows.append(
                {
                    "kind": kind,
                    "input_tokens": len(tokens),
                    "file": str(destination),
                    "memory_before": before,
                    "memory_after": memory(),
                }
            )
            report["runs"].append({"length": length, "trial": trial, **rows[-1]})
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(label + " complete", flush=True)
report["complete"] = True
args.output.write_text(json.dumps(report, indent=2) + "\n")
