# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
python = str(root / ".host-venv/bin/python")
base = "http://127.0.0.1:18111"
deadline = time.monotonic() + 7200
for marker in ["NATIVE_DSO_ADMISSION.json", "accounting-checkpoint.json"]:
    while not (out / marker).exists():
        assert time.monotonic() < deadline
        time.sleep(10)
images = {
    "safe": json.loads((out / "bsafe-next/manifest.json").read_text())["image_id"],
    "new": json.loads((out / "native-lt-image/manifest.json").read_text())["image_id"],
    "prev": json.loads((out / "BPREV_CONTINUATION.json").read_text())["bprev_image"],
}
benchmark_source = root / "benchmarks/benchmark_v152.py"
benchmark_sha = hashlib.sha256(benchmark_source.read_bytes()).hexdigest()
active_name = None
active_folder = None
monitor = None
monitor_output = None


def post(path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        return json.loads(data) if data else None


def rpc(method, **kwargs):
    return post("/collective_rpc", {"method": method, "kwargs": kwargs, "timeout": 120})


def save(name, value):
    (active_folder / name).write_text(json.dumps(value, indent=2) + "\n")


def benchmark(label, concurrency, lengths, repeats=2, output_tokens=1024):
    assert hashlib.sha256(benchmark_source.read_bytes()).hexdigest() == benchmark_sha
    path = active_folder / (label + ".json")
    assert not path.exists()
    command = [
        python,
        "benchmarks/benchmark_v152.py",
        "--base",
        base,
        "--output",
        str(path),
        "--lengths",
        *map(str, lengths),
        "--repeats",
        str(repeats),
        "--output-tokens",
        str(output_tokens),
        "--concurrency",
        str(concurrency),
    ]
    started = time.time()
    with (active_folder / (label + ".log")).open("w") as log:
        subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    result = json.loads(path.read_text())
    assert result["complete"]
    for row in result["runs"]:
        assert row["emitted_tokens"] == concurrency * output_tokens
        assert all(len(item["tokens"]) == output_tokens for item in row["results"])
    save(
        label + ".stage.json",
        {
            "command": command,
            "benchmark_sha256": benchmark_sha,
            "started": started,
            "finished": time.time(),
            "process": json.loads((active_folder / "process.json").read_text()),
        },
    )
    print(
        json.dumps({"process": active_name, "stage": label, "complete": True}),
        flush=True,
    )
    return result


def launch(label, arm):
    global active_name, active_folder, monitor, monitor_output
    active_folder = out / label
    assert not active_folder.exists(), (
        "Preserve prior process artifacts before a restart"
    )
    active_folder.mkdir(parents=True)
    active_name = "1cat-v152-next-" + label.replace("/", "-")
    command = json.loads((out / "bsafe-service-command.json").read_text())
    command[command.index("--name") + 1] = active_name
    command[command.index("--worker-extension-cls") + 1] = (
        "next_qualification_worker.QualificationWorker"
    )
    original = images["safe"]
    index = command.index(original)
    for i, value in enumerate(command):
        if value.endswith(":/profiles"):
            profiles = active_folder / "profiles"
            profiles.mkdir()
            command[i] = str(profiles) + ":/profiles"
    command[index:index] = [
        "-v",
        str(out / "incoming/next_qualification_worker.py")
        + ":/opt/venv/lib/python3.12/site-packages/next_qualification_worker.py:ro",
    ]
    command[command.index(original)] = images[arm]
    save("launch-command.json", command)
    subprocess.run(command, cwd=root, check=True)
    process = json.loads(
        subprocess.check_output(["docker", "inspect", active_name], text=True)
    )[0]
    save(
        "process.json",
        {
            "container_id": process["Id"],
            "started_at": process["State"]["StartedAt"],
            "image_id": process["Image"],
            "arm": arm,
            "independent_unit": "Fresh service process",
            "benchmark_sha256": benchmark_sha,
        },
    )
    monitor_output = (active_folder / "gpu-process-monitor.log").open("w")
    monitor = subprocess.Popen(
        ["nvidia-smi", "pmon", "-s", "um", "-d", "1", "-o", "DT"],
        stdout=monitor_output,
        stderr=subprocess.STDOUT,
    )
    ready_deadline = time.monotonic() + 1200
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                assert response.status == 200
            break
        except OSError:
            running = subprocess.check_output(
                ["docker", "inspect", "-f", "{{.State.Running}}", active_name],
                text=True,
            ).strip()
            assert running == "true" and time.monotonic() < ready_deadline
            time.sleep(10)
    config = rpc("v152_runtime_configuration")
    for row in config["results"]:
        assert row["max_num_seqs"] == 8 and row["async_scheduling"]
        assert row["cache"]["block_size"] == 2048
        assert row["cache"]["mamba_block_size"] == 8192
        assert row["cache"]["prefix_cache_retention_interval"] == 0
        assert not row["cache"]["partial_prefix_cache"]
    save("startup-configuration.json", config)
    save("startup-identity.json", rpc("longctx_identity"))
    print("Ready " + active_name, flush=True)


def stop():
    global active_name, active_folder, monitor, monitor_output
    if active_name is not None:
        with (active_folder / "server.log").open("w") as log:
            subprocess.run(
                ["docker", "logs", active_name], stdout=log, stderr=subprocess.STDOUT
            )
        subprocess.run(["docker", "stop", "--timeout", "60", active_name], check=True)
    if monitor is not None:
        monitor.terminate()
        monitor.wait(timeout=30)
        monitor_output.close()
    active_name = None
    active_folder = None
    monitor = None
    monitor_output = None


def warmup():
    for concurrency in [1, 2, 3]:
        benchmark("startup-c" + str(concurrency), concurrency, [1024], 1, 128)
    before = rpc("v152_runtime_configuration")
    assert all(row["rejection_graphs"] >= 3 for row in before["results"])
    save("startup-sampler-graphs.json", before)
    benchmark("warm-c8", 8, [1024], 1, 128)
    benchmark("warm-long", 1, [32768], 1, 32)
    precision = rpc("v152_precision_and_heads")
    for row in precision["results"]:
        value = row["matmul_precision"]["allow_fp16_reduced_precision_reduction"]
        assert value is False or (isinstance(value, list) and value[0] is False)
    save("precision-and-heads.json", precision)
    save("warm-identity.json", rpc("longctx_identity"))


try:
    # Route/profile and cancellation canaries use their own disposable process.
    launch("service-canary", "new")
    warmup()
    save("profile-start.json", rpc("v152_start_profile"))
    benchmark("route-proof-cold131k", 1, [131072], 1, 64)
    profile = rpc("v152_stop_profile", label="cold131k-route-proof")
    save("profile-stop.json", profile)
    counts = []
    for row in profile["results"]:
        path = active_folder / "profiles" / Path(row["trace"]).name
        events = json.loads(path.read_text())["traceEvents"]
        owned = [
            event
            for event in events
            if event.get("name") == "sm70_nvfp4_prefill_lt" and event.get("ph") == "X"
        ]
        counts.append(
            {
                "rank": row["rank"],
                "fixed_policy_calls": len(owned),
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    assert len(counts) == 4 and all(row["fixed_policy_calls"] > 0 for row in counts)
    save(
        "actual-route-owner.json",
        {
            "complete": True,
            "counts": counts,
            "scope": (
                "One complete cold131K request, normal configuration. Profile "
                "durations are diagnostic only."
            ),
        },
    )
    with (active_folder / "cancel-state.log").open("w") as log:
        subprocess.run(
            [
                python,
                "benchmarks/functional_v152_cancel_state.py",
                "--output",
                str(active_folder / "cancel-state.json"),
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    cancellation = json.loads((active_folder / "cancel-state.json").read_text())
    assert cancellation["complete"] and cancellation["passed"]
    save(
        "complete.json",
        {
            "complete": True,
            "image_id": images["new"],
            "route_counts": counts,
            "cancellation_passed": True,
        },
    )
    stop()

    # Bprev bookends measure correctness cost. Middle four processes are ABBA.
    sequence = [
        ("prev-p1", "prev"),
        ("safe-a1", "safe"),
        ("new-b1", "new"),
        ("new-b2", "new"),
        ("safe-a2", "safe"),
        ("prev-p2", "prev"),
    ]
    processes = []
    for label, arm in sequence:
        launch("formal/" + label, arm)
        warmup()
        if arm == "prev":
            benchmark("formal-c1", 1, [1024, 32768, 131072])
            benchmark("formal-c8", 8, [1024])
        else:
            benchmark("formal-c1", 1, [1024, 4096, 32768, 131072, 261120])
            benchmark("formal-c2", 2, [1024, 4096, 32768])
            benchmark("formal-c4", 4, [1024, 4096, 32768])
            benchmark("formal-c8", 8, [1024, 32768])
        save("final-identity.json", rpc("longctx_identity"))
        save("final-memory.json", rpc("longctx_memory"))
        process = json.loads((active_folder / "process.json").read_text())
        save(
            "complete.json",
            {
                "complete": True,
                "process": process,
                "formal_output_tokens": 1024,
                "repeats_per_cell": 2,
                "cache": (
                    "Prefix reset before each formal batch, JIT/graphs warmed. GPU "
                    "L2 is not globally flushed."
                ),
            },
        )
        processes.append({"label": label, "arm": arm, **process})
        stop()
        (out / "FORMAL_SERVICE_PROGRESS.json").write_text(
            json.dumps(processes, indent=2) + "\n"
        )
    (out / "FORMAL_SERVICE_ADMISSION.json").write_text(
        json.dumps(
            {
                "complete": True,
                "images": images,
                "processes": processes,
                "ordering": "Bprev bookends with Bsafe/Bnew ABBA inside",
                "independent_units_per_arm": {"prev": 2, "safe": 2, "new": 2},
                "scope": (
                    "Trace-off service measurements. Related quality, independent "
                    "async/cache efficiency and final qualification remain pending."
                ),
            },
            indent=2,
        )
        + "\n"
    )
finally:
    stop()
