# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the recorded contract from a relocated, verified evidence bundle."""

import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--image-id", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--models", type=Path, default=Path("/data/models"))
parser.add_argument("--profile", choices=["main", "cache", "async"], default="main")
args = parser.parse_args()
bundle = args.artifacts.resolve()
source = args.source.resolve()
assert args.image_id.startswith("sha256:") and len(args.image_id) == 71
actual = subprocess.check_output(
    ["docker", "image", "inspect", "-f", "{{.Id}}", args.image_id], text=True
).strip()
assert actual == args.image_id
assert (source / "benchmarks/v152_worker_probe.py").is_file()
assert (bundle / "raw/fixtures/chat_template.jinja").is_file()
args.output.mkdir(parents=True, exist_ok=False)
cache = args.output.resolve() / "cache"
profiles = args.output.resolve() / "profiles"
cache.mkdir()
profiles.mkdir()
command = json.loads((bundle / "raw/bsafe-service-command.json").read_text())
frozen = json.loads((bundle / "raw/bsafe-next/manifest.json").read_text())["image_id"]
command[command.index("--name") + 1] = args.name
command[command.index("--worker-extension-cls") + 1] = (
    "next_qualification_worker.QualificationWorker"
)
for index, value in enumerate(command):
    if value == "/data/models:/data/models:ro":
        command[index] = str(args.models.resolve()) + ":/data/models:ro"
    elif value.endswith(":/root/.cache"):
        command[index] = str(cache) + ":/root/.cache"
    elif value.endswith(":/templates:ro"):
        command[index] = str(bundle / "raw/fixtures") + ":/templates:ro"
    elif value.endswith(":/profiles"):
        command[index] = str(profiles) + ":/profiles"
    elif value.endswith(
        ":/opt/venv/lib/python3.12/site-packages/v152_worker_probe.py:ro"
    ):
        command[index] = (
            str(source / "benchmarks/v152_worker_probe.py")
            + ":/opt/venv/lib/python3.12/site-packages/v152_worker_probe.py:ro"
        )
position = command.index(frozen)
command[position:position] = [
    "-v",
    str(bundle / "raw/incoming/next_qualification_worker.py")
    + ":/opt/venv/lib/python3.12/site-packages/next_qualification_worker.py:ro",
]
command[command.index(frozen)] = args.image_id
if args.profile == "cache":
    command += ["--prefix-cache-retention-interval", "8192", "--partial-prefix-cache"]
if args.profile == "async":
    selection = json.loads((bundle / "raw/ASYNC_SELECTION.json").read_text())
    assert selection["retain_optional_preset"], (
        "Use a profile admitted in ASYNC_SELECTION.json"
    )
    chunk = str(selection["selected_chunk"])
    command[command.index("--max-num-batched-tokens") + 1] = chunk
    command[command.index("--long-prefill-token-threshold") + 1] = chunk
    command += [
        "--mixed-prefill-step-latency-ms",
        "250",
        "--mixed-prefill-min-tokens",
        str(selection["selected_min_tokens"]),
    ]
(args.output / "launch-command.json").write_text(json.dumps(command, indent=2) + "\n")
subprocess.run(command, check=True)
print(
    json.dumps(
        {
            "container": args.name,
            "image_id": args.image_id,
            "health_url": "http://127.0.0.1:18111/health",
            "next": (
                "Wait for health, then warm C1/C2/C3 through benchmark_v152.py "
                "before formal measurement."
            ),
        }
    ),
    flush=True,
)
