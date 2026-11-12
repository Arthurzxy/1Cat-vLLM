# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch an isolated cache profile, with optional per-group diagnostics."""

import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=["off", "retained", "default"], required=True)
parser.add_argument("--arm", default="best-source")
parser.add_argument("--no-trace", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = root / "bench_results/v152_opt_20261111"
suffix = "-formal" if args.no_trace else ""
label = "cache-" + args.profile + suffix
command = json.loads((out / (args.arm + "-command.json")).read_text())
command[command.index("--name") + 1] = "1cat-v152-opt-" + label
image = json.loads((out / args.arm / "manifest.json").read_text())["image_id"]

if not args.no_trace:
    trace = out / (label + "-trace")
    trace.mkdir(exist_ok=True)
    trigger = out / "v152_cache_diagnostic.pth"
    trigger.write_text("import v152_cache_diagnostic\n")
    options = [
        "-e",
        "V152_CACHE_TRACE=/cache-trace",
        "-v",
        str(trace) + ":/cache-trace",
        "-v",
        str(root / "benchmarks/v152_cache_diagnostic.py")
        + ":/opt/venv/lib/python3.12/site-packages/v152_cache_diagnostic.py:ro",
        "-v",
        str(trigger)
        + ":/opt/venv/lib/python3.12/site-packages/v152_cache_diagnostic.pth:ro",
    ]
    index = command.index(image)
    command[index:index] = options

if args.profile == "retained":
    command += ["--prefix-cache-retention-interval", "8192", "--partial-prefix-cache"]
if args.profile == "default":
    for flag in ["--block-size", "--mamba-block-size"]:
        index = command.index(flag)
        del command[index : index + 2]

(out / (label + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
subprocess.run(command, check=True)
