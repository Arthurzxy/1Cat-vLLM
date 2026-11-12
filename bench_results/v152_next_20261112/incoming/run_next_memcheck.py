# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
deadline = time.monotonic() + 86400
while not (out / "EFFICIENCY_SERVICE_RUNS_COMPLETE.json").exists():
    assert time.monotonic() < deadline
    time.sleep(10)
image = json.loads((out / "native-lt-image/manifest.json").read_text())["image_id"]
script = out / "incoming/next_native_memcheck_smoke.py"
command = [
    "docker",
    "run",
    "--name",
    "1cat-v152-next-native-memcheck",
    "--gpus",
    "device=6",
    "--ipc=host",
    "--network",
    "none",
    "-v",
    str(root) + ":/workspace",
    "-v",
    "/data/models:/data/models:ro",
    "-w",
    "/tmp",
    "--entrypoint",
    "/usr/local/cuda/compute-sanitizer/compute-sanitizer",
    image,
    "--tool",
    "memcheck",
    "--error-exitcode",
    "99",
    "/workspace/.venv/bin/python",
    "/workspace/" + str(script.relative_to(root)),
]
log = out / "native-memcheck.log"
with log.open("w") as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
assert "ERROR SUMMARY: 0 errors" in log.read_text()
report = json.loads((out / "NATIVE_MEMCHECK_SMOKE.json").read_text())
assert report["complete"] and len(report["rows"]) == 6
report.update(
    tool="compute-sanitizer memcheck",
    error_count=0,
    image_id=image,
    command=command,
    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
)
(out / "NATIVE_MEMCHECK_ADMISSION.json").write_text(json.dumps(report, indent=2) + "\n")
print("Six actual native shapes passed memcheck and output-byte checks", flush=True)
