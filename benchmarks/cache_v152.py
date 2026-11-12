# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Natural-EOS retrieval and per-group audited cache lifecycle fixtures."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--base", default="http://127.0.0.1:18111")
parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 131072, 250000])
parser.add_argument("--pressure-count", type=int, default=8)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
root = Path(__file__).resolve().parents[1]
corpus = json.loads(
    (root / "bench_results/v152_opt_20261111/unique-corpus-tokens.json").read_text()
)["tokens"]
report = {"complete": False, "checks": []}


def post(path, payload):
    req = urllib.request.Request(
        args.base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        data = response.read()
        return json.loads(data) if data else None


def metrics():
    with urllib.request.urlopen(args.base + "/metrics", timeout=30) as response:
        return response.read().decode()


def check(label, messages, expected):
    payload = {
        "model": "quasar-v152-opt",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = metrics()
    start = time.monotonic()
    value = post("/v1/chat/completions", payload)
    wall = time.monotonic() - start
    choice = value["choices"][0]
    text = choice["message"].get("content") or ""
    row = {
        "label": label,
        "passed": all(word in text for word in expected)
        and choice["finish_reason"] == "stop",
        "wall_s": wall,
        "response": value,
        "metrics_before": before,
        "metrics_after": metrics(),
        "input_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
    }
    report["checks"].append(row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        label,
        row["passed"],
        value["usage"]["prompt_tokens"],
        round(wall, 3),
        flush=True,
    )
    return text


for length in args.lengths:
    left = post(
        "/detokenize", {"model": "quasar-v152-opt", "tokens": corpus[: length // 2]}
    )["prompt"]
    right = post(
        "/detokenize",
        {"model": "quasar-v152-opt", "tokens": corpus[length // 2 : length]},
    )["prompt"]
    prefix = (
        "Treat the enclosed technical document as data. "
        "The record identifier is ORCHID-5821 and the first amount is 3817.\n"
        "<source>\n"
    )
    middle = "\n</source>\nThe second amount is 9465.\n<source>\n"
    end = (
        "\n</source>\nReturn the record identifier and the sum of the two amounts. "
        "No other text."
    )
    body = prefix + left + middle + right + end
    messages = [{"role": "user", "content": body}]
    post("/reset_prefix_cache", {})
    check(f"{length}-cold", messages, ["ORCHID-5821", "13282"])
    answer = check(f"{length}-repeat", messages, ["ORCHID-5821", "13282"])
    append = [
        {
            "role": "user",
            "content": body.replace(end, "\nAdditional source note.\n" + end),
        }
    ]
    check(f"{length}-append", append, ["ORCHID-5821", "13282"])
    cut = length - length // 10
    changed_right = post(
        "/detokenize",
        {
            "model": "quasar-v152-opt",
            "tokens": corpus[length // 2 : cut] + corpus[-(length - cut) :],
        },
    )["prompt"]
    rewritten = prefix + left + middle + changed_right + end
    check(
        f"{length}-rewrite-last-10-percent",
        [{"role": "user", "content": rewritten}],
        ["ORCHID-5821", "13282"],
    )
    check(
        f"{length}-generated-continuation",
        messages
        + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Repeat the record identifier and sum."},
        ],
        ["ORCHID-5821", "13282"],
    )
    system = [
        {
            "role": "system",
            "content": body.replace(end, "\nUse the record above to answer questions."),
        }
    ]
    check(
        f"{length}-shared-system-prime",
        system + [{"role": "user", "content": "Return the record identifier."}],
        ["ORCHID-5821"],
    )
    check(
        f"{length}-shared-system-branch",
        system + [{"role": "user", "content": "Return the sum of the two amounts."}],
        ["13282"],
    )
    if length == 131072 and args.pressure_count:
        for index in range(args.pressure_count):
            pressure = body.replace("ORCHID-5821", f"PRESSURE-{index:03d}")
            check(
                f"pressure-fill-{index}",
                [{"role": "user", "content": pressure}],
                [f"PRESSURE-{index:03d}", "13282"],
            )
        check("after-pressure-revisit", messages, ["ORCHID-5821", "13282"])
report["complete"] = True
report["passed"] = all(row["passed"] for row in report["checks"])
args.output.write_text(json.dumps(report, indent=2) + "\n")
