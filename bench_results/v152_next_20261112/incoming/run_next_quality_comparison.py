# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import time
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
deadline = time.monotonic() + 86400
while not (out / "QUALITY_SERVICE_RUNS_COMPLETE.json").exists():
    assert time.monotonic() < deadline
    time.sleep(10)
subprocess.run(
    [
        str(root / ".host-venv/bin/python"),
        str(out / "incoming/compare_next_quality.py"),
        "--root",
        str(out),
    ],
    check=True,
)
