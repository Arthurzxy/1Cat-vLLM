# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional, cancellation, ragged-batch and prefix regression fixtures."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import struct
import time
import urllib.request
import zlib
from pathlib import Path

import pybase64 as base64
from v152_validation import has_generated_text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--base", default="http://127.0.0.1:18111")
parser.add_argument("--prefix-length", type=int, default=32768)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
report = {"complete": False, "checks": []}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(args.output)


def post(path, payload):
    request = urllib.request.Request(
        args.base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        data = response.read()
        return json.loads(data) if data else None


def chat(content, **extra):
    payload = {
        "model": "quasar-v152-opt",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 512,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra)
    start = time.perf_counter()
    value = post("/v1/chat/completions", payload)
    return {
        "response": value,
        "wall_s": time.perf_counter() - start,
        "input_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
    }


def text(row):
    return row["response"]["choices"][0]["message"].get("content") or ""


def add(label, passed, value):
    report["checks"].append({"label": label, "passed": bool(passed), "result": value})
    args.output.write_text(json.dumps(report, indent=2) + "\n")


for temperature in (0.0, 0.7, 1.0, 0.7):
    value = chat(
        "What is 17 times 23? Return only the integer.",
        temperature=temperature,
        top_k=20,
        top_p=0.95,
    )
    add("live-temperature-" + str(temperature), text(value).strip() == "391", value)
value = chat(
    "Calculate 17 times 23 and explain briefly.",
    chat_template_kwargs={"enable_thinking": True},
    max_tokens=1024,
)
add(
    "thinking",
    "391" in text(value) and value["response"]["choices"][0]["finish_reason"] == "stop",
    value,
)
value = chat(
    "Calculate 17 times 23. Return it as answer.",
    temperature=0.7,
    top_k=20,
    top_p=0.95,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
)
try:
    valid = json.loads(text(value)) == {"answer": 391}
except ValueError:
    valid = False
add("structured-output", valid, value)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
value = chat(
    "Use get_weather to look up the weather in London.", tools=tools, tool_choice="auto"
)
calls = value["response"]["choices"][0]["message"].get("tool_calls") or []
add("tool-parser", bool(calls) and calls[0]["function"]["name"] == "get_weather", value)


def chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress((b"\x00" + bytes([255, 0, 0]) * 32) * 32))
    + chunk(b"IEND", b"")
)
image = "data:image/png;base64," + base64.b64encode(png).decode()
value = chat(
    [
        {"type": "text", "text": "What color fills this image? Answer one word."},
        {"type": "image_url", "image_url": {"url": image}},
    ]
)
add("vision-red", "red" in text(value).lower(), value)
video_path = root / "bench_results/v152_opt_20261111/red-2frames.mp4"
video = "data:video/mp4;base64," + base64.b64encode(video_path.read_bytes()).decode()
value = chat(
    [
        {"type": "text", "text": "What color dominates this video? Answer one word."},
        {"type": "video_url", "video_url": {"url": video}},
    ]
)
add("video-red", "red" in text(value).lower(), value)
value = chat(
    "What is 17 times 23? Return only the integer.", logprobs=True, top_logprobs=5
)
entries = (value["response"]["choices"][0].get("logprobs") or {}).get("content") or []
add(
    "generated-logprobs",
    bool(entries) and all(math.isfinite(row["logprob"]) for row in entries),
    value,
)
for index in range(12):
    payload = {
        "model": "quasar-v152-opt",
        "messages": [
            {"role": "user", "content": "List the integers from 1 through 500."}
        ],
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.95,
        "seed": index,
        "max_tokens": 1024,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    observed_generation = False
    with urllib.request.urlopen(req, timeout=120) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if has_generated_text(json.loads(line[6:])):
                observed_generation = True
                break
    marker = f"FRESH_{index:02d}"
    value = chat("Return exactly " + marker + ". No other text.", max_tokens=64)
    value["cancel_observed_nonempty_text"] = observed_generation
    add(
        "cancel-slot-reuse-" + str(index),
        observed_generation and marker in text(value),
        value,
    )


def ragged(index):
    marker = f"REQUEST_{index:03d}"
    value = chat(
        "Write "
        + marker
        + ", list the integers from 1 through "
        + str(20 + index % 5 * 15)
        + ", then repeat "
        + marker
        + ".",
        temperature=(0.0, 0.7, 1.0)[index % 3],
        top_k=20,
        top_p=0.95,
        max_tokens=384,
        seed=index,
    )
    return marker, value


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for marker, value in pool.map(ragged, range(24)):
        add("ragged-refill-" + marker, marker in text(value), value)
fixture = json.loads(
    (root / "bench_results/v152_opt_20261111/unique-corpus-tokens.json").read_text()
)
parts = []
for name, digest in fixture["sources"].items():
    data = (root / name).read_text()
    if hashlib.sha256(data.encode()).hexdigest() != digest:
        raise RuntimeError("Frozen source changed: " + name)
    parts.append(data)
raw_ids = post("/tokenize", {"model": "quasar-v152-opt", "prompt": "\n".join(parts)})[
    "tokens"
]
body = post(
    "/detokenize", {"model": "quasar-v152-opt", "tokens": raw_ids[: args.prefix_length]}
)["prompt"]
start = "The document identifier is ORCHID-5821. Remember it.\n"
question = "\nReturn only the document identifier."
post("/reset_prefix_cache", {})
for label, content in [
    ("prefix-prime", start + body + question),
    ("prefix-repeat", start + body + question),
    (
        "prefix-append",
        start + body + "\nAdditional note about checking evidence." + question,
    ),
    (
        "prefix-rewrite-tail",
        start
        + body[: int(len(body) * 0.9)]
        + "\nThe remaining passage has been revised.\n"
        + body[-100:]
        + question,
    ),
]:
    value = chat(content, max_tokens=64)
    add(label, "ORCHID-5821" in text(value), value)
report["complete"] = True
report["passed"] = all(row["passed"] for row in report["checks"])
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps({"passed": report["passed"], "checks": len(report["checks"])}),
    flush=True,
)
