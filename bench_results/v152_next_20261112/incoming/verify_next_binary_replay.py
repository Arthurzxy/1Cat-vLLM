# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild binary overlays and compare every installed vllm runtime file."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
assert json.loads((out / "EFFICIENCY_SERVICE_RUNS_COMPLETE.json").read_text())[
    "complete"
]
context = root / "replay-context-next-20261112"
context.mkdir(exist_ok=True)
archive = root / "bench_results/v152_opt_20261111/delivery/v152-opt-artifacts.tar.gz"
with tarfile.open(archive, "r|gz") as source:
    for info in source:
        name = info.name.removeprefix("v152-opt-artifacts/")
        if (
            not (name.startswith("python/") or name.startswith("native/"))
            or info.isdir()
        ):
            continue
        assert info.isfile() and ".." not in PurePosixPath(name).parts
        target = context / "previous" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.extractfile(info).read())
        target.chmod(info.mode)
for name in [
    "Dockerfile.replay-bprev",
    "Dockerfile.replay-bsafe",
    "Dockerfile.replay-bnew",
    ".dockerignore",
]:
    shutil.copy2(out / name, context / name)
for folder in ["bsafe-next", "native-lt-image"]:
    shutil.copytree(
        out / folder / "image-context",
        context / "raw" / folder / "image-context",
        dirs_exist_ok=True,
    )
python = """import hashlib
import json
import stat
from pathlib import Path
root = Path('/opt/venv/lib/python3.12/site-packages/vllm')
files = {}
for path in sorted(root.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
        files[str(path.relative_to(root))] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'mode': stat.S_IMODE(path.stat().st_mode),
        }
print(json.dumps(files, sort_keys=True))
"""


def runtime(image):
    return json.loads(
        subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/opt/venv/bin/python",
                image,
                "-c",
                python,
            ],
            text=True,
        )
    )


original = {
    "bprev": json.loads((out / "BPREV_CONTINUATION.json").read_text())["bprev_image"],
    "bsafe": json.loads((out / "bsafe-next/manifest.json").read_text())["image_id"],
    "bnew": json.loads((out / "native-lt-image/manifest.json").read_text())["image_id"],
}
environment = os.environ.copy()
environment["DOCKER_BUILDKIT"] = "0"
results = {}
for label, image in original.items():
    tag = "1cat-v152-next-replay-qa:" + label
    log_path = out / ("binary-replay-" + label + ".log")
    if log_path.exists():
        log_path.rename(
            out / ("binary-replay-" + label + "-prior-" + str(time.time_ns()) + ".log")
        )
    with log_path.open("w") as log:
        subprocess.run(
            [
                "docker",
                "build",
                "--pull=false",
                "--network=none",
                "-f",
                "Dockerfile.replay-" + label,
                "-t",
                tag,
                ".",
            ],
            cwd=context,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    reconstructed = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", tag], text=True
    ).strip()
    left = runtime(image)
    right = runtime(reconstructed)
    assert left == right, label
    configs = [
        json.loads(
            subprocess.check_output(["docker", "image", "inspect", value], text=True)
        )[0]["Config"]
        for value in [image, reconstructed]
    ]
    for key in [
        "Env",
        "Entrypoint",
        "Cmd",
        "User",
        "WorkingDir",
        "ExposedPorts",
        "Volumes",
        "Healthcheck",
    ]:
        assert configs[0].get(key) == configs[1].get(key), (label, key)
    results[label] = {
        "original_image": image,
        "replay_image": reconstructed,
        "runtime_files_compared": len(left),
        "all_bytes_and_modes_equal": True,
        "runtime_inventory_sha256": hashlib.sha256(
            json.dumps(left, sort_keys=True).encode()
        ).hexdigest(),
        "configuration_equal": True,
    }
(out / "BINARY_REPLAY_QA.json").write_text(
    json.dumps(
        {
            "complete": True,
            "passed": True,
            "images": results,
            "context_files_sha256": {
                str(path.relative_to(context)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(context.rglob("*"))
                if path.is_file()
            },
            "scope": (
                "CPU-only reconstruction. Every installed vllm file byte and mode "
                "is compared, excluding derived pycache. Model quality "
                "qualification belongs to the original images with identical "
                "runtime content."
            ),
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(results, indent=2), flush=True)
