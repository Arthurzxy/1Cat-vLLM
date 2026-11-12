# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream a resident long decoder while admitting 32K and short prefills."""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--base", default="http://127.0.0.1:18111")
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
corpus = json.loads(
    (root / "bench_results/v152_opt_20261111/unique-corpus-tokens.json").read_text()
)["tokens"]
if args.output.exists():
    raise FileExistsError(args.output)
report = {"complete": False, "trials": []}
salts = {}
for index in (
    list(range(12))
    + list(range(100, 100 + args.repeats))
    + list(range(200, 200 + args.repeats))
):
    req = urllib.request.Request(
        args.base + "/tokenize",
        data=json.dumps(
            {"model": "quasar-v152-opt", "prompt": f"Mixed fixture {index:04d}.\n"}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        salts[index] = json.load(response)["tokens"]


def metrics():
    with urllib.request.urlopen(args.base + "/metrics", timeout=30) as response:
        return response.read().decode()


def request(label, length, output, index, ready=None):
    salt = salts[index]
    ids = corpus[:32] + salt + corpus[32 : length - 64 - len(salt)] + corpus[-64:]
    assert len(ids) == length
    payload = {
        "model": "quasar-v152-opt",
        "prompt": ids,
        "max_tokens": output,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.95,
        "seed": index,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    req = urllib.request.Request(
        args.base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    chunks = []
    tokens = []
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                new = choice.get("token_ids") or []
                if new:
                    chunks.append(
                        {"at_s": time.monotonic() - start, "tokens": len(new)}
                    )
                    tokens.extend(new)
                    if ready:
                        ready.set()
    assert (
        usage["prompt_tokens"] == length
        and len(tokens) == output == usage["completion_tokens"]
    )
    return {
        "label": label,
        "start": start,
        "end": time.monotonic(),
        "ttft_s": chunks[0]["at_s"],
        "usage": usage,
        "chunks": chunks,
        "tokens": tokens,
    }


for trial in range(args.repeats):
    reset = urllib.request.Request(
        args.base + "/reset_prefix_cache",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(reset, timeout=60).close()
    before = metrics()
    ready = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        long = pool.submit(request, "resident-long", 131072, 2048, 100 + trial, ready)
        if not ready.wait(300):
            raise TimeoutError("long decoder did not begin")
        inserted = time.monotonic()
        futures = [pool.submit(request, "insert-32k", 32768, 512, 200 + trial)]
        for index in range(12):
            futures.append(
                pool.submit(request, "short-" + str(index), 1024, 128, index)
            )
            time.sleep(0.1)
        rows = [long.result()] + [future.result() for future in futures]
    resident = rows[0]
    assert resident["start"] + resident["ttft_s"] <= inserted < resident["end"]
    report["trials"].append(
        {
            "inserted": inserted,
            "requests": rows,
            "metrics_before": before,
            "metrics_after": metrics(),
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("completed mixed trial", trial, flush=True)
report["complete"] = True
args.output.write_text(json.dumps(report, indent=2) + "\n")
