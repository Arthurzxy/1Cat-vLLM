# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only telemetry across quality, efficiency and sanitizer stages."""

import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

out = Path("/home/llm/1cat-v152-opt-20261111/bench_results/v152_next_20261112")
deadline = time.monotonic() + 86400
while not (out / "SHORT_CONFIRM_ADMISSION.json").exists():
    assert time.monotonic() < deadline
    time.sleep(5)
commands = {
    "gpu-telemetry-continuous.csv": [
        "nvidia-smi",
        "-i",
        "6,7,8,9",
        (
            "--query-gpu=timestamp,index,uuid,pci.bus_id,name,pstate,clocks.current"
            ".sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,m"
            "emory.used"
        ),
        "--format=csv",
        "--loop=1",
    ],
    "gpu-processes-continuous.log": [
        "nvidia-smi",
        "pmon",
        "-i",
        "6,7,8,9",
        "-s",
        "um",
        "-d",
        "1",
        "-o",
        "DT",
    ],
}
handles = []
processes = []
started = time.time()
try:
    for name, command in commands.items():
        stream = (out / name).open("w")
        handles.append(stream)
        processes.append(
            subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        )
    while not (out / "NATIVE_MEMCHECK_ADMISSION.json").exists():
        assert time.monotonic() < deadline
        assert all(process.poll() is None for process in processes)
        time.sleep(5)
finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=30)
    for stream in handles:
        stream.close()
assert (out / "NATIVE_MEMCHECK_ADMISSION.json").exists()
(out / "RESOURCE_MONITOR_COMPLETE.json").write_text(
    json.dumps(
        {
            "complete": True,
            "started_epoch": started,
            "finished_epoch": time.time(),
            "host_timezone": str(datetime.datetime.now().astimezone().tzinfo),
            "commands": commands,
            "files_sha256": {
                name: hashlib.file_digest((out / name).open("rb"), "sha256").hexdigest()
                for name in commands
            },
            "scope": (
                "Continuous1Hz GPU6-9 telemetry and per-process activity covering "
                "quality, independent efficiency and native memcheck. Earlier main "
                "and short-confirm studies retain their per-process pmon logs and "
                "request endpoint GPU snapshots."
            ),
        },
        indent=2,
    )
    + "\n"
)
print("Resource monitoring completed and streams closed", flush=True)
