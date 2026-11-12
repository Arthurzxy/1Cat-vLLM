# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Close task-owned resources after every queued GPU validation completes."""

import json
import subprocess
import time
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
deadline = time.monotonic() + 86400
while not all(
    (out / name).exists()
    for name in ["REBUILD_BOOTSTRAP_QA.json", "REBUILD_TOOLCHAIN_QA.json"]
):
    assert time.monotonic() < deadline
    time.sleep(10)
assert json.loads((out / "REBUILD_BOOTSTRAP_QA.json").read_text())["passed"]
assert json.loads((out / "REBUILD_TOOLCHAIN_QA.json").read_text())["passed"]
for name in [
    "EFFICIENCY_SUMMARY.json",
    "NATIVE_MEMCHECK_ADMISSION.json",
    "RESOURCE_MONITOR_COMPLETE.json",
]:
    assert json.loads((out / name).read_text())["complete"], name
quality = json.loads((out / "FINAL_QUALITY_SUMMARY.json").read_text())
assert quality["passed"]
running = subprocess.check_output(
    ["docker", "ps", "--format", "{{.Names}}"], text=True
).splitlines()
stopped = []
for name in running:
    if name.startswith("1cat-v152-next-"):
        subprocess.run(["docker", "stop", "--time", "60", name], check=True)
        stopped.append(name)
assert not [
    name
    for name in subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}}"], text=True
    ).splitlines()
    if name.startswith("1cat-v152-next-")
]
prior = json.loads((out / "BPREV_CONTINUATION.json").read_text())["services"]
services = {}
for name, original in prior.items():
    result = subprocess.run(["docker", "inspect", name], text=True, capture_output=True)
    current = json.loads(result.stdout)[0] if result.returncode == 0 else None
    services[name] = {"original": original, "exists": current is not None}
    if current:
        services[name].update(
            {
                "id": current["Id"],
                "image": current["Image"],
                "started_at": current["State"]["StartedAt"],
                "running": current["State"]["Running"],
                "identity_preserved": original["id"] == current["Id"]
                and original["image"] == current["Image"]
                and original["started_at"] == current["State"]["StartedAt"],
            }
        )
listeners = subprocess.check_output(["ss", "-H", "-ltn", "sport = :18111"], text=True)
assert not listeners.strip(), listeners
gpu = subprocess.check_output(
    [
        "nvidia-smi",
        (
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,"
            "utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw"
        ),
        "--format=csv",
    ],
    text=True,
)
pmon = subprocess.check_output(
    ["nvidia-smi", "pmon", "-i", "6,7,8,9", "-c", "1"], text=True
)
result = {
    "complete": True,
    "finished_epoch_s": time.time(),
    "task_owned_containers_stopped": stopped,
    "existing_services": services,
    "all_original_service_identities_preserved": all(
        row.get("identity_preserved") and row.get("running")
        for row in services.values()
    ),
    "local_test_port_18111_free": True,
    "gpu": gpu,
    "pmon": pmon,
    "resource_scope": (
        "GPU6-9 experiment processes exited. Shared services retain their own GPU "
        "contexts. Task images, caches, source and evidence remain available."
    ),
}
(out / "CLEANUP.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            key: value
            for key, value in result.items()
            if key not in ["existing_services", "gpu", "pmon"]
        },
        indent=2,
    )
)
