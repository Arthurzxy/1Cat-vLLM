# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the narrow original-cache counterfactual on immutable Bsafe-next."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
folder = out / "cache-original-window-logic"
folder.mkdir(exist_ok=True)
assert not (folder / "manifest.json").exists()
parent = json.loads((out / "bsafe-next/manifest.json").read_text())["image_id"]
context = folder / "image-context"
files = {}
for name in [
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
]:
    data = subprocess.check_output(
        ["git", "show", "2bb8c2b9cd7c1696630b6be464052027f0dd0147:" + name], cwd=root
    )
    destination = context / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    files[name] = hashlib.sha256(data).hexdigest()
(context / "Dockerfile").write_text(
    "FROM " + parent + "\nCOPY vllm/ /opt/venv/lib/python3.12/site-packages/vllm/\n"
)
tag = "1cat-vllm:v152-next-cache-original-window-logic-20261112"
with (folder / "build.log").open("w") as log:
    subprocess.run(
        ["docker", "build", "--pull=false", "--network=none", "-t", tag, str(context)],
        env={**os.environ, "DOCKER_BUILDKIT": "0"},
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
identity = subprocess.check_output(
    ["docker", "image", "inspect", "-f", "{{.Id}}", tag], text=True
).strip()
(folder / "manifest.json").write_text(
    json.dumps(
        {
            "image_id": identity,
            "image_tag": tag,
            "parent_image_id": parent,
            "files": files,
            "scope": (
                "Only two cache lookup/window Python files at B0. Resample guard "
                "and all other Bsafe repairs and native binaries remain inherited."
            ),
        },
        indent=2,
    )
    + "\n"
)
print(identity, flush=True)
