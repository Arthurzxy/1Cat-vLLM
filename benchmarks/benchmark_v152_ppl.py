# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WikiText-2 prompt NLL plus fixed continuations after long prefixes."""

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path


def post(base, path, payload, timeout=1800):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def nll(tokens, rows, begin=1):
    if len(tokens) != len(rows):
        raise ValueError("Prompt-logprob length mismatch")
    values = []
    for index in range(max(1, begin), len(tokens)):
        row = rows[index]
        if row is None or str(tokens[index]) not in row:
            raise ValueError(f"Missing actual-token logprob at {index}")
        values.append(-float(row[str(tokens[index])]["logprob"]))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Missing or non-finite NLL")
    return sum(values), len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18111")
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    text = args.text.read_text()
    tokens = post(args.base, "/tokenize", {"model": "quasar-v152-opt", "prompt": text})[
        "tokens"
    ]
    report = {
        "text_sha256": hashlib.sha256(args.text.read_bytes()).hexdigest(),
        "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "cases": [],
        "complete": False,
    }

    def score(ids):
        response = post(
            args.base,
            "/v1/completions",
            {
                "model": "quasar-v152-opt",
                "prompt": ids,
                "max_tokens": 1,
                "temperature": 0,
                "seed": 0,
                "prompt_logprobs": 0,
            },
            timeout=1800,
        )
        rows = response["choices"][0]["prompt_logprobs"]
        raw = args.output.with_suffix(".rows")
        raw.mkdir(exist_ok=True)
        import gzip

        with gzip.open(raw / f"{len(report['cases']):04d}.json.gz", "wt") as stream:
            json.dump({"tokens": ids, "prompt_logprobs": rows}, stream)
        return rows

    total, count = 0.0, 0
    for start in range(0, len(tokens), 1024):
        ids = tokens[start : start + 1024]
        if len(ids) < 2:
            continue
        value, size = nll(ids, score(ids))
        total += value
        count += size
        report["cases"].append(
            {"kind": "wikitext", "start": start, "nll_sum": value, "count": size}
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if len(report["cases"]) % 25 == 0:
            print(f"PPL chunks {len(report['cases'])}", flush=True)
    report["ppl"] = math.exp(total / count)
    report["n_tokens_scored"] = count
    for length in (8192, 32768, 65536, 131072, 250000):
        salt = post(
            args.base,
            "/tokenize",
            {"model": "quasar-v152-opt", "prompt": f"Continuation fixture {length}.\n"},
        )["tokens"]
        body = (tokens * (length // len(tokens) + 2))[: length - len(salt)]
        ids = salt + body
        value, size = nll(ids, score(ids), begin=length - 512)
        report["cases"].append(
            {"kind": "continuation", "length": length, "nll_sum": value, "count": size}
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Continuation {length} mean NLL {value / size}", flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
