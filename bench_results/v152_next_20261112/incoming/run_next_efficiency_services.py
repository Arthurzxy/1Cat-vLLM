# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial independent cache and async experiments after kernel quality runs."""

import hashlib
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
old = root / "bench_results/v152_opt_20261111"
python = str(root / ".host-venv/bin/python")
incoming = out / "incoming"
base = "http://127.0.0.1:18111"
deadline = time.monotonic() + 86400
while not (out / "QUALITY_SERVICE_RUNS_COMPLETE.json").exists():
    assert time.monotonic() < deadline
    time.sleep(10)
decision = json.loads((out / "SERVICE_WINNER_DECISION.json").read_text())
safe = json.loads((out / "bsafe-next/manifest.json").read_text())["image_id"]
new = json.loads((out / "native-lt-image/manifest.json").read_text())["image_id"]
selected = new if decision["retain_native_candidate"] else safe
counterfactual = json.loads(
    (out / "cache-original-window-logic/manifest.json").read_text()
)["image_id"]
(incoming / "next_scheduler_observer.pth").write_text(
    "import next_scheduler_observer\n"
)


def post(path, value):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        return json.loads(data) if data else None


def rpc(method, **kwargs):
    return post("/collective_rpc", {"method": method, "kwargs": kwargs, "timeout": 120})


def execute(folder, label, command):
    with (folder / (label + ".log")).open("w") as log:
        subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    (folder / (label + ".stage.json")).write_text(
        json.dumps(
            {
                "command": command,
                "finished": time.time(),
                "source_sha256": {
                    value: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                    for value in command
                    if value.endswith(".py") and Path(value).is_file()
                },
                "complete": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(folder.name + "/" + label + " complete", flush=True)


def benchmark(folder, label, concurrency=1, lengths=(1024,), output=128):
    execute(
        folder,
        label,
        [
            python,
            str(root / "benchmarks/benchmark_v152.py"),
            "--output",
            str(folder / (label + ".json")),
            "--lengths",
            *map(str, lengths),
            "--repeats",
            "1",
            "--output-tokens",
            str(output),
            "--concurrency",
            str(concurrency),
        ],
    )


def run_service(
    label,
    image,
    body,
    *,
    interval=0,
    partial=False,
    chunk=8192,
    floor=0,
    scheduler_trace=False,
    cache_trace=False,
):
    folder = out / "efficiency" / label
    folder.mkdir(parents=True, exist_ok=False)
    name = "1cat-v152-next-efficiency-" + label
    command = json.loads((out / "bsafe-service-command.json").read_text())
    command[command.index("--name") + 1] = name
    command[command.index("--worker-extension-cls") + 1] = (
        "next_efficiency_worker.EfficiencyWorker"
    )
    command[command.index("--max-num-batched-tokens") + 1] = str(chunk)
    command[command.index("--long-prefill-token-threshold") + 1] = str(chunk)
    position = command.index(safe)
    for i, value in enumerate(command):
        if value.endswith(":/profiles"):
            profiles = folder / "profiles"
            profiles.mkdir()
            command[i] = str(profiles) + ":/profiles"
    extra = []
    for module in ["next_efficiency_worker", "next_qualification_worker"]:
        extra += [
            "-v",
            str(incoming / (module + ".py"))
            + ":/opt/venv/lib/python3.12/site-packages/"
            + module
            + ".py:ro",
        ]
    if scheduler_trace:
        trace = folder / "scheduler-trace"
        trace.mkdir()
        extra += [
            "-e",
            "V152_SCHEDULER_TRACE=/scheduler-trace",
            "-v",
            str(trace) + ":/scheduler-trace",
        ]
        for suffix in ["py", "pth"]:
            extra += [
                "-v",
                str(incoming / ("next_scheduler_observer." + suffix))
                + ":/opt/venv/lib/python3.12/site-packages/next_scheduler_observer."
                + suffix
                + ":ro",
            ]
    if cache_trace:
        trace = folder / "cache-trace"
        trace.mkdir()
        extra += [
            "-e",
            "V152_CACHE_TRACE=/cache-trace",
            "-v",
            str(trace) + ":/cache-trace",
            "-v",
            str(old / "cache-ablation-observer.py")
            + ":/opt/venv/lib/python3.12/site-packages/v152_cache_diagnostic.py:ro",
            "-v",
            str(old / "cache-ablation-observer.pth")
            + ":/opt/venv/lib/python3.12/site-packages/v152_cache_diagnostic.pth:ro",
        ]
    command[position:position] = extra
    command[command.index(safe)] = image
    command += [
        "--prefix-cache-retention-interval",
        str(interval),
        "--mixed-prefill-step-latency-ms",
        "250",
        "--mixed-prefill-min-tokens",
        str(floor),
    ]
    if partial:
        command.append("--partial-prefix-cache")
    (folder / "launch-command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    try:
        startup_deadline = time.monotonic() + 1200
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as response:
                    assert response.status == 200
                break
            except OSError:
                running = subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.State.Running}}", name], text=True
                ).strip()
                assert running == "true" and time.monotonic() < startup_deadline
                time.sleep(10)
        process = json.loads(
            subprocess.check_output(["docker", "inspect", name], text=True)
        )[0]
        configuration = rpc("v152_efficiency_configuration")
        for row in configuration["results"]:
            scheduler = row["scheduler"]
            assert scheduler["async_scheduling"] and scheduler["max_num_seqs"] == 8
            assert scheduler["max_num_batched_tokens"] == chunk
            assert scheduler["mixed_prefill_step_latency_ms"] == 250
            assert scheduler["mixed_prefill_min_tokens"] == floor
            assert row["cache"]["prefix_cache_retention_interval"] == interval
            assert row["cache"]["partial_prefix_cache"] == partial
        (folder / "identity.json").write_text(
            json.dumps(
                {
                    "configuration": configuration,
                    "container_id": process["Id"],
                    "image": process["Image"],
                    "started": process["State"]["StartedAt"],
                    "native": rpc("longctx_identity"),
                    "instrumented": scheduler_trace or cache_trace,
                },
                indent=2,
            )
            + "\n"
        )
        for concurrency in [1, 2, 3]:
            benchmark(folder, "startup-c" + str(concurrency), concurrency)
        body(folder)
        (folder / "final-memory.json").write_text(
            json.dumps(rpc("longctx_memory"), indent=2) + "\n"
        )
        (folder / "complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "image": image,
                    "configuration": configuration,
                    "instrumented": scheduler_trace or cache_trace,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        with (folder / "server.log").open("w") as log:
            subprocess.run(
                ["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT
            )
        subprocess.run(["docker", "stop", "--timeout", "60", name], check=True)
    return folder


def cache_ablation(folder):
    execute(
        folder,
        "chat",
        [
            python,
            str(root / "benchmarks/cache_v152.py"),
            "--output",
            str(folder / "chat.json"),
            "--lengths",
            "32768",
            "--pressure-count",
            "0",
        ],
    )
    execute(
        folder,
        "fixed",
        [
            python,
            str(incoming / "next_cache_fixed_client.py"),
            "--output",
            str(folder / "fixed.json"),
            "--lengths",
            "32768",
            "--repeats",
            "1",
        ],
    )


cache_profiles = [
    ("original-window-logic", 8192, True, counterfactual),
    ("partial-only", 0, True, safe),
    ("checkpoint-only", 8192, False, safe),
    ("combined", 8192, True, safe),
]
for label, interval, partial, image in cache_profiles:
    run_service(
        "cache-" + label,
        image,
        cache_ablation,
        interval=interval,
        partial=partial,
        cache_trace=True,
    )
(out / "CACHE_FOUR_ARM_COMPLETE.json").write_text(
    json.dumps(
        {
            "complete": True,
            "scope": (
                "Four independently started Bsafe cache attribution arms. All "
                "timing is instrumented diagnostic."
            ),
            "profiles": [row[0] for row in cache_profiles],
        },
        indent=2,
    )
    + "\n"
)


def mixed(folder, repeats=3):
    execute(
        folder,
        "mixed",
        [
            python,
            str(root / "benchmarks/mixed_v152.py"),
            "--output",
            str(folder / "mixed.json"),
            "--repeats",
            str(repeats),
        ],
    )


run_service(
    "async-age-diagnostic", safe, lambda folder: mixed(folder, 1), scheduler_trace=True
)
profiles = [
    ("default8192-a1", 8192, 0),
    ("chunk4096-b1", 4096, 0),
    ("floor4096-c1", 8192, 4096),
]
for label, chunk, floor in profiles:
    run_service(label, safe, mixed, chunk=chunk, floor=floor)


def p95(values):
    values = sorted(values)
    return values[int(0.95 * (len(values) - 1) + 0.999999)]


def mixed_summary(label):
    value = json.loads((out / "efficiency" / label / "mixed.json").read_text())
    assert value["complete"]
    short = []
    long_wall = []
    gaps = []
    inserted_ttft = []
    throughput = []
    for trial in value["trials"]:
        requests = trial["requests"]
        resident = requests[0]
        long_wall.append(resident["end"] - resident["start"])
        chunks = [resident["start"] + chunk["at_s"] for chunk in resident["chunks"]]
        gaps.extend(b - a for a, b in zip(chunks, chunks[1:]) if a >= trial["inserted"])
        short.extend(
            row["ttft_s"] for row in requests if row["label"].startswith("short-")
        )
        inserted_ttft.extend(
            row["ttft_s"] for row in requests if row["label"] == "insert-32k"
        )
        throughput.append(
            sum(len(row["tokens"]) for row in requests)
            / (
                max(row["end"] for row in requests)
                - min(row["start"] for row in requests)
            )
        )
    return {
        "short_empirical_p95_s": p95(short),
        "short_count": len(short),
        "resident_mean_wall_s": statistics.mean(long_wall),
        "resident_chunk_gap_p95_s": p95(gaps),
        "resident_max_chunk_gap_s": max(gaps),
        "insert32k_mean_ttft_s": statistics.mean(inserted_ttft),
        "whole_workload_mean_output_tps": statistics.mean(throughput),
        "independent_processes": 1,
        "trials": len(value["trials"]),
    }


screen = {label: mixed_summary(label) for label, _, _ in profiles}
baseline = screen["default8192-a1"]
winner = min(
    profiles[1:], key=lambda profile: screen[profile[0]]["short_empirical_p95_s"]
)
candidate = screen[winner[0]]
promising = (
    candidate["short_empirical_p95_s"] < baseline["short_empirical_p95_s"] * 0.95
    and candidate["resident_max_chunk_gap_s"]
    <= baseline["resident_max_chunk_gap_s"] * 1.05
    and candidate["insert32k_mean_ttft_s"] <= baseline["insert32k_mean_ttft_s"] * 1.05
    and candidate["whole_workload_mean_output_tps"]
    >= baseline["whole_workload_mean_output_tps"] * 0.95
)
(out / "ASYNC_SCREEN.json").write_text(
    json.dumps(
        {
            "complete": True,
            "screen": screen,
            "promising": promising,
            "candidate": winner,
            "selection": (
                "At least5% lower short empirical p95 with at most5% observed "
                "regressions in max resident chunk gap, inserted32K TTFT and whole "
                "workload output throughput."
            ),
        },
        indent=2,
    )
    + "\n"
)
if promising:
    _, chunk, floor = winner
    run_service("async-selected-b2", safe, mixed, chunk=chunk, floor=floor)
    run_service("default8192-a2", safe, mixed)
    screen["async-selected-b2"] = mixed_summary("async-selected-b2")
    screen["default8192-a2"] = mixed_summary("default8192-a2")
    second = screen["async-selected-b2"]
    second_baseline = screen["default8192-a2"]
    promising = (
        second["short_empirical_p95_s"]
        < second_baseline["short_empirical_p95_s"] * 0.95
        and second["resident_max_chunk_gap_s"]
        <= second_baseline["resident_max_chunk_gap_s"] * 1.05
        and second["insert32k_mean_ttft_s"]
        <= second_baseline["insert32k_mean_ttft_s"] * 1.05
        and second["whole_workload_mean_output_tps"]
        >= second_baseline["whole_workload_mean_output_tps"] * 0.95
    )
(out / "ASYNC_SELECTION.json").write_text(
    json.dumps(
        {
            "complete": True,
            "retain_optional_preset": promising,
            "selected_chunk": winner[1] if promising else 8192,
            "selected_min_tokens": winner[2] if promising else 0,
            "results": screen,
            "limits": (
                "Empirical p95 from36 short requests/process. Process-level "
                "confirmation only for a promising candidate. No p99 or statistical"
                " no-regression guarantee."
            ),
        },
        indent=2,
    )
    + "\n"
)


def final_cache(folder):
    execute(
        folder,
        "fixed",
        [
            python,
            str(incoming / "next_cache_fixed_client.py"),
            "--output",
            str(folder / "fixed.json"),
            "--lengths",
            "131072",
            "250000",
            "--repeats",
            "2",
        ],
    )
    execute(
        folder,
        "chat-pressure",
        [
            python,
            str(root / "benchmarks/cache_v152.py"),
            "--output",
            str(folder / "chat-pressure.json"),
            "--lengths",
            "131072",
            "--pressure-count",
            "8",
        ],
    )
    benchmark(folder, "cache-c8-32k", 8, (32768,), 1024)
    execute(
        folder,
        "cancellation",
        [
            python,
            str(root / "benchmarks/functional_v152_cancel_state.py"),
            "--output",
            str(folder / "cancel-state.json"),
        ],
    )


run_service(
    "final-cache-combination", selected, final_cache, interval=8192, partial=True
)
if promising:

    def final_async(folder):
        mixed(folder)
        execute(
            folder,
            "functional",
            [
                python,
                str(root / "benchmarks/functional_v152.py"),
                "--output",
                str(folder / "functional.json"),
            ],
        )
        execute(
            folder,
            "cancellation",
            [
                python,
                str(root / "benchmarks/functional_v152_cancel_state.py"),
                "--output",
                str(folder / "cancel-state.json"),
            ],
        )

    run_service(
        "final-async-combination",
        selected,
        final_async,
        chunk=winner[1],
        floor=winner[2],
    )
(out / "EFFICIENCY_SERVICE_RUNS_COMPLETE.json").write_text(
    json.dumps(
        {
            "complete": True,
            "selected_kernel_image": selected,
            "optional_async_preset": promising,
            "scope": (
                "Raw serial cache, async and final combination runs complete. "
                "Paired analysis and final qualification pending."
            ),
        },
        indent=2,
    )
    + "\n"
)
