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
old = root / "bench_results/v152_opt_20261111"
python = str(root / ".host-venv/bin/python")
base = "http://127.0.0.1:18111"
deadline = time.monotonic() + 10800
while not (out / "SERVICE_WINNER_DECISION.json").exists():
    assert time.monotonic() < deadline
    time.sleep(10)
decision = json.loads((out / "SERVICE_WINNER_DECISION.json").read_text())
assert decision["formal_measurements_complete"]
images = {
    "safe": json.loads((out / "bsafe-next/manifest.json").read_text())["image_id"],
    "new": json.loads((out / "native-lt-image/manifest.json").read_text())["image_id"],
}
arms = ["safe", "new"] if decision["retain_native_candidate"] else ["safe"]
grader = subprocess.check_output(
    ["docker", "image", "inspect", "-f", "{{.Id}}", "1cat-vllm:whl-v1.5.2-2bb8c2b9"],
    text=True,
).strip()
assert grader.startswith("sha256:")
gates = {
    "quality": (
        "Candidate grade count per category is at least the Bsafe-next count on the"
        " same1819 frozen cases."
    ),
    "ppl_relative_change_max": 1e-6,
    "nll_absolute_change_max": 1e-5,
    "functional": (
        "Compare all50 existing fixture verdicts and preserve common marker failures."
    ),
    "cancellation": (
        "Actual physical-slot reuse and clean-reference output/state must pass."
    ),
    "grader_image_id": grader,
    "quality_cases_sha256": hashlib.sha256(
        (old / "quality-data/cases.json").read_bytes()
    ).hexdigest(),
    "wikitext_sha256": hashlib.sha256(
        (old / "quality-data/wikitext-2-raw-test.txt").read_bytes()
    ).hexdigest(),
}
(out / "FINAL_QUALITY_GATES.json").write_text(json.dumps(gates, indent=2) + "\n")


def rpc(method, **kwargs):
    request = urllib.request.Request(
        base + "/collective_rpc",
        data=json.dumps({"method": method, "kwargs": kwargs, "timeout": 120}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def execute(folder, label, command):
    record = {
        "command": command,
        "started": time.time(),
        "script_sha256": {
            str(Path(value).relative_to(root)): hashlib.sha256(
                Path(value).read_bytes()
            ).hexdigest()
            for value in command
            if value.endswith(".py") and Path(value).is_file()
        },
    }
    with (folder / (label + ".log")).open("w") as log:
        subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    record.update(complete=True, finished=time.time())
    (folder / (label + ".stage.json")).write_text(json.dumps(record, indent=2) + "\n")
    print("Completed quality stage " + folder.name + "/" + label, flush=True)


summaries = []
for arm in arms:
    folder = out / ("quality-" + arm)
    folder.mkdir(exist_ok=True)
    name = "1cat-v152-next-quality-" + arm
    command = json.loads((out / "bsafe-service-command.json").read_text())
    command[command.index("--name") + 1] = name
    command[command.index("--worker-extension-cls") + 1] = (
        "next_qualification_worker.QualificationWorker"
    )
    index = command.index(images["safe"])
    profiles = folder / "profiles"
    profiles.mkdir(exist_ok=True)
    for i, value in enumerate(command):
        if value.endswith(":/profiles"):
            command[i] = str(profiles) + ":/profiles"
    command[index:index] = [
        "-v",
        str(out / "incoming/next_qualification_worker.py")
        + ":/opt/venv/lib/python3.12/site-packages/next_qualification_worker.py:ro",
    ]
    command[command.index(images["safe"])] = images[arm]
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    try:
        ready_deadline = time.monotonic() + 1200
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as response:
                    assert response.status == 200
                break
            except OSError:
                running = subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.State.Running}}", name], text=True
                ).strip()
                assert running == "true" and time.monotonic() < ready_deadline
                time.sleep(10)
        for concurrency in [1, 2, 3]:
            execute(
                folder,
                "startup-c" + str(concurrency),
                [
                    python,
                    str(root / "benchmarks/benchmark_v152.py"),
                    "--output",
                    str(folder / ("startup-c" + str(concurrency) + ".json")),
                    "--lengths",
                    "1024",
                    "--repeats",
                    "1",
                    "--output-tokens",
                    "128",
                    "--concurrency",
                    str(concurrency),
                    "--base",
                    base,
                ],
            )
        for method in [
            "longctx_identity",
            "v152_runtime_configuration",
            "v152_precision_and_heads",
        ]:
            (folder / (method + ".json")).write_text(
                json.dumps(rpc(method), indent=2) + "\n"
            )
        execute(
            folder,
            "functional",
            [
                python,
                str(root / "benchmarks/functional_v152.py"),
                "--output",
                str(folder / "functional.json"),
                "--base",
                base,
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
                "--base",
                base,
            ],
        )
        cancel = json.loads((folder / "cancel-state.json").read_text())
        assert cancel["complete"] and cancel["passed"]
        execute(
            folder,
            "quality",
            [
                python,
                str(out / "incoming/run_next_quality_client.py"),
                "--grader-image",
                grader,
                "--cases",
                str(old / "quality-data/cases.json"),
                "--output",
                str(folder / "mbpp-gsm8k"),
                "--base",
                base,
                "--image-id",
                images[arm],
            ],
        )
        execute(
            folder,
            "ppl",
            [
                python,
                str(root / "benchmarks/benchmark_v152_ppl.py"),
                "--text",
                str(old / "quality-data/wikitext-2-raw-test.txt"),
                "--output",
                str(folder / "ppl.json"),
                "--base",
                base,
            ],
        )
        ppl = json.loads((folder / "ppl.json").read_text())
        functional = json.loads((folder / "functional.json").read_text())
        assert ppl["complete"] and functional["complete"]
        (folder / "final-memory.json").write_text(
            json.dumps(rpc("longctx_memory"), indent=2) + "\n"
        )
        summary = {
            "arm": arm,
            "image_id": images[arm],
            "complete": True,
            "quality": json.loads((folder / "mbpp-gsm8k/summary.json").read_text()),
            "ppl_file": str(folder / "ppl.json"),
            "functional_file": str(folder / "functional.json"),
        }
        (folder / "complete.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
    finally:
        with (folder / "server.log").open("w") as log:
            subprocess.run(
                ["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT
            )
        subprocess.run(["docker", "stop", "--timeout", "60", name], check=True)
(out / "QUALITY_SERVICE_RUNS_COMPLETE.json").write_text(
    json.dumps(
        {
            "complete": True,
            "arms": summaries,
            "gates": gates,
            "status": (
                "Raw runs complete, paired quality/NLL/verdict comparison pending"
            ),
        },
        indent=2,
    )
    + "\n"
)
