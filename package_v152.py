# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a task-local image with an explicit source and native manifest."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", required=True)
parser.add_argument("--source-build", action="store_true")
parser.add_argument("--baseline-python", action="store_true")
parser.add_argument("--fa2-library", type=Path)
parser.add_argument("--core-library", type=Path)
args = parser.parse_args()
root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_opt_20261111" / args.name
out.mkdir(parents=True, exist_ok=True)
if (out / "manifest.json").exists():
    raise FileExistsError(out / "manifest.json")
context = Path(tempfile.mkdtemp(prefix="image-context-", dir=out))
(context / "vllm").mkdir()
paths = subprocess.check_output(
    ["git", "diff", "HEAD", "--name-only", "--", "vllm"], cwd=root, text=True
).splitlines()
if args.baseline_python:
    paths = []
files = {}
for name in paths:
    path = root / name
    destination = context / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
dockerfile = [
    "FROM 1cat-vllm:whl-v1.5.2-2bb8c2b9",
    "COPY vllm/ /opt/venv/lib/python3.12/site-packages/vllm/",
]
if args.source_build:
    prefix = root / "bench_results/v152_opt_20261111/bsrc-fa2/install/vllm"
    for relative in (
        "_C.abi3.so",
        "_C_stable_libtorch.abi3.so",
        "_sm70_sampler_C.abi3.so",
        "vllm_flash_attn/_vllm_fa2_C.abi3.so",
    ):
        path = prefix / relative
        if args.fa2_library and relative.startswith("vllm_flash_attn"):
            path = args.fa2_library
        if args.core_library and relative == "_C.abi3.so":
            path = args.core_library
        destination = context / "vllm" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        files["vllm/" + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
(context / "Dockerfile").write_text("\n".join(dockerfile) + "\n")
tag = "1cat-vllm:v152-opt-" + args.name
command = ["docker", "build", "-t", tag, str(context)]
with (out / "build.log").open("w") as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
identity = subprocess.check_output(
    ["docker", "image", "inspect", "--format", "{{.Id}}", tag], text=True
).strip()
manifest = {
    "image": tag,
    "image_id": identity,
    "files": files,
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "dirty_patch": ""
    if args.baseline_python
    else subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True),
    "native_scope": "Rebuilt _C, _C_stable_libtorch, _sm70_sampler_C and FA2"
    if args.source_build
    else "Verified v1.5.2 wheel native libraries, Python ABI unchanged",
    "command": command,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(identity)
