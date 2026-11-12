# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and install the standard SM70 FA2 component into an isolated prefix."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--jobs", type=int, default=4)
parser.add_argument("--triton-source", type=Path)
parser.add_argument("--source-root", type=Path)
parser.add_argument("--target", choices=("_vllm_fa2_C", "_C"), default="_vllm_fa2_C")
args = parser.parse_args()
checkout = Path(__file__).resolve().parents[1]
root = args.source_root.resolve() if args.source_root else checkout
out = args.run_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
if (out / "manifest.json").exists():
    raise FileExistsError(out / "manifest.json")
inputs = {}
for directory in ("csrc", "cmake"):
    for path in sorted((root / directory).rglob("*")):
        if path.is_file():
            inputs[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
for name in (
    "CMakeLists.txt",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/config/flash_v100.py",
):
    inputs[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(out / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
env = os.environ.copy()
env["PATH"] = str(checkout / ".venv/bin") + os.pathsep + env["PATH"]
env["TORCH_CUDA_ARCH_LIST"] = "7.0"
cmake = str(checkout / ".venv/bin/cmake")
commands = [
    [
        cmake,
        "-S",
        str(root),
        "-B",
        str(out / "build"),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DVLLM_TARGET_DEVICE=cuda",
        f"-DVLLM_PYTHON_EXECUTABLE={sys.executable}",
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
        "-DVLLM_FLASH_ATTN_SM70=ON",
        "-DVLLM_SM70_79T_PREFILL=ON",
        f"-DFETCHCONTENT_BASE_DIR={out / 'deps'}",
        f"-DCMAKE_INSTALL_PREFIX={out / 'install'}",
        "-DNVCC_THREADS=1",
    ],
    [
        cmake,
        "--build",
        str(out / "build"),
        "--target",
        args.target,
        "-j",
        str(args.jobs),
    ],
    [cmake, "--install", str(out / "build"), "--component", args.target],
]
dependency_identity = {}
if args.triton_source:
    triton_source = args.triton_source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(triton_source), "rev-parse", "HEAD"], text=True
    ).strip()
    pinned = subprocess.check_output(
        ["git", "-C", str(triton_source), "rev-parse", "v3.5.1"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(triton_source), "status", "--porcelain"], text=True
    ).strip()
    if commit != pinned or dirty:
        raise ValueError("Reused Triton source must match clean pinned v3.5.1")
    commands[0].append(f"-DFETCHCONTENT_SOURCE_DIR_TRITON_KERNELS={triton_source}")
    dependency_identity["triton"] = {"path": str(triton_source), "commit": commit}
(out / "dependency-identity.json").write_text(
    json.dumps(dependency_identity, indent=2) + "\n"
)
(out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
for index, command in enumerate(commands):
    with (out / f"stage-{index}.log").open("w") as log:
        subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
libraries = list((out / "install").rglob(args.target + ".abi3.so"))
if len(libraries) != 1:
    raise RuntimeError(f"Expected one installed component library, got {libraries}")
for name, digest in inputs.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Build input changed: {name}")
library = libraries[0]
manifest = {
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "library": str(library),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "complete": True,
    "target": args.target,
    "scope": "standard CMake component, model admission pending",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
