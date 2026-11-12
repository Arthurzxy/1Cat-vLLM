# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired streaming fixtures with unique repository text and raw service metrics."""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import regex as re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", default="http://127.0.0.1:18111")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096])
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--output-tokens", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--concurrency", type=int, default=1)
parser.add_argument("--natural-eos", action="store_true")
parser.add_argument("--tokens-file", type=Path)
parser.add_argument("--keep-prefix-cache", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / "bench_results/v152_opt_20261111"
model = "quasar-v152-opt"


def post(path, payload):
    request = urllib.request.Request(
        args.base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        data = response.read()
        return json.loads(data) if data else None


def metrics():
    with urllib.request.urlopen(args.base + "/metrics", timeout=30) as response:
        body = response.read().decode()
    values = {}
    for line in body.splitlines():
        if line.startswith("#") or not line:
            continue
        name, value = line.rsplit(" ", 1)
        key = name.split("{", 1)[0]
        if (
            "spec_decode" not in key
            and not key.endswith(
                (
                    "_time_seconds_sum",
                    "_latency_seconds_sum",
                    "_computed_tokens_sum",
                    "_cache_hits_total",
                    "_preemptions_total",
                )
            )
            and key not in ("vllm:num_requests_running", "vllm:num_requests_waiting")
        ):
            continue
        position = re.search(r'position="(\d+)"', name)
        if position:
            key += "/position=" + position[1]
        values[key] = values.get(key, 0) + float(value)
    return values


def gpu():
    import subprocess

    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,"
            "temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    )


def request(ids, seed):
    payload = {
        "model": model,
        "prompt": ids,
        "max_tokens": args.output_tokens,
        "temperature": args.temperature,
        "top_k": 20,
        "top_p": 0.95,
        "seed": seed,
        "ignore_eos": not args.natural_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    tokens, text, chunks = [], [], []
    usage = None
    finish = None
    start = time.perf_counter()
    req = urllib.request.Request(
        args.base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
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
                ids_new = choice.get("token_ids") or []
                content = choice.get("text") or ""
                if ids_new or content:
                    chunks.append(
                        {"at_s": time.perf_counter() - start, "tokens": len(ids_new)}
                    )
                tokens.extend(ids_new)
                text.append(content)
                finish = choice.get("finish_reason") or finish
    assert usage and usage["completion_tokens"] == len(tokens)
    assert len(tokens) <= args.output_tokens
    if not args.natural_eos:
        assert len(tokens) == args.output_tokens
    assert usage["prompt_tokens"] == len(ids)
    wall = time.perf_counter() - start
    return {
        "prompt_tokens": len(ids),
        "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "usage": usage,
        "tokens": tokens,
        "text": "".join(text),
        "chunks": chunks,
        "finish": finish,
        "wall_s": wall,
        "ttft_s": chunks[0]["at_s"],
        "finished_monotonic_s": time.perf_counter(),
        "http_output_tps": len(tokens) / wall,
        "max_repeated_token_run": max(
            (len(list(group)) for _, group in __import__("itertools").groupby(tokens)),
            default=0,
        ),
    }


fixture = args.tokens_file or out / "unique-corpus-tokens.json"
if args.tokens_file is not None and not fixture.is_file():
    raise FileNotFoundError(fixture)
if not fixture.exists():
    documents = []
    size = 0
    sources = {}
    paths = sorted((root / "docs").rglob("*.md")) + sorted(
        (root / "vllm").rglob("*.py")
    )
    for path in paths:
        data = path.read_text()
        relative = str(path.relative_to(root))
        documents.append("\nFile " + relative + "\n" + data)
        sources[relative] = hashlib.sha256(data.encode()).hexdigest()
        size += len(data)
        if size > 2_000_000:
            break
    text = (
        "Read the technical source material below. After reading, write a "
        "detailed explanation of its main implementation techniques.\n"
        + "".join(documents)
        + "\nExplain the code and its numerical behavior."
    )
    encoded = post(
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "add_generation_prompt": True,
        },
    )
    fixture.write_text(json.dumps({"tokens": encoded["tokens"], "sources": sources}))
corpus = json.loads(fixture.read_text())["tokens"]
assert len(corpus) >= max(args.lengths), len(corpus)
report = {
    "complete": False,
    "config": vars(args)
    | {
        "output": str(args.output),
        "tokens_file": str(args.tokens_file) if args.tokens_file else None,
    },
    "source_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "runs": [],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(args.output)
for length in args.lengths:
    ids = corpus[: length - 64] + corpus[-64:]
    inputs = [ids]
    if args.concurrency > 1:
        inputs = []
        for index in range(args.concurrency):
            salt = post(
                "/tokenize",
                {"model": model, "prompt": f"Concurrent fixture identifier {index}.\n"},
            )["tokens"]
            inputs.append(ids[:32] + salt + ids[32 : -64 - len(salt)] + ids[-64:])
            assert len(inputs[-1]) == length
    for trial in range(args.repeats):
        if not args.keep_prefix_cache:
            post("/reset_prefix_cache", {})
        before = metrics()
        gpu_before = gpu()
        started = time.perf_counter()
        residency = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [
                pool.submit(request, inputs[index], 20261111 + trial + index)
                for index in range(args.concurrency)
            ]
            while any(not future.done() for future in futures):
                sample = metrics()
                residency.append(
                    {
                        "at_s": time.perf_counter() - started,
                        "running": sample.get("vllm:num_requests_running"),
                        "waiting": sample.get("vllm:num_requests_waiting"),
                    }
                )
                time.sleep(0.25)
            results = [future.result() for future in futures]
        elapsed = max(result["finished_monotonic_s"] for result in results) - started
        after = metrics()
        delta = {key: after.get(key, 0) - before.get(key, 0) for key in before | after}
        row = {
            "length": length,
            "trial": trial,
            "results": results,
            "metrics": delta,
            "gpu_before": gpu_before,
            "gpu_after": gpu(),
            "residency": residency,
            "emitted_tokens": sum(len(result["tokens"]) for result in results),
            "batch_wall_s": elapsed,
            "aggregate_output_tps": sum(len(result["tokens"]) for result in results)
            / elapsed,
        }
        report["runs"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "length": length,
                    "trial": trial,
                    "ttft_s": [value["ttft_s"] for value in results],
                    "aggregate_output_tps": row["aggregate_output_tps"],
                    "accepted": {
                        key: value
                        for key, value in delta.items()
                        if "spec_decode" in key
                    },
                }
            ),
            flush=True,
        )
report["complete"] = True
args.output.write_text(json.dumps(report, indent=2) + "\n")
