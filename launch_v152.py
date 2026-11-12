# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--max-num-seqs", type=int, default=4)
parser.add_argument("--diagnostics", action="store_true")
parser.add_argument("--rpc-profile", action="store_true")
args = parser.parse_args()
root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_opt_20261111"
manifest = json.loads((out / args.arm / "manifest.json").read_text())
command = json.loads((out / "bwheel-command.json").read_text())
if not args.diagnostics:
    diagnostic = command.index("VLLM_FLASH_V100_ROUTE_SUMMARY=1")
    del command[diagnostic - 1 : diagnostic + 1]
command[command.index("--max-num-seqs") + 1] = str(args.max_num_seqs)
command[command.index("--name") + 1] = "1cat-v152-opt-" + args.arm
index = command.index("1cat-vllm:whl-v1.5.2-2bb8c2b9")
command[index] = manifest["image_id"]
command[command.index(str(out / "bwheel-cache") + ":/root/.cache")] = (
    str(out / (args.arm + "-cache")) + ":/root/.cache"
)
options = [
    "-v",
    f"{root}/benchmarks/v152_worker_probe.py:"
    "/opt/venv/lib/python3.12/site-packages/v152_worker_probe.py:ro",
]
if args.profile or args.rpc_profile:
    profiles = out / (args.arm + "-profiles")
    profiles.mkdir(exist_ok=True)
    options += ["-v", f"{profiles}:/profiles"]
command[index:index] = options
command += ["--worker-extension-cls", "v152_worker_probe.LongctxWorkerProbe"]
if args.profile:
    command += [
        "--profiler-config",
        json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": "/profiles",
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
                "ignore_frontend": True,
                "max_iterations": 12,
            }
        ),
    ]
(out / (args.arm + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
print(subprocess.check_output(command, text=True).strip())
