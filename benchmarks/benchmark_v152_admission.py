# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect resumable full MBPP/GSM8K results with isolated code execution."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import regex as re
from v152_validation import final_answer

IMAGE = "1cat-vllm:whl-v1.5.2-2bb8c2b9"


def grade(case, text, finish_reason):
    if finish_reason != "stop":
        return {"passed": False, "reason": "truncated_or_non_stop"}
    body = final_answer(text)
    if case["category"] == "gsm8k":
        answers = re.findall(r"####\s*([-+]?\d[\d,.]*)", body)
        answer = answers[-1].replace(",", "").rstrip(".") if answers else None
        return {
            "passed": answer == case["answer"],
            "answer": answer,
            "expected": case["answer"],
        }
    fence = chr(96) * 3
    fenced = re.findall(fence + r"(?:python|py)?\s*\n(.*?)" + fence, body, re.S)
    code = max(fenced, key=len) if fenced else body
    try:
        ast.parse(code)
    except SyntaxError:
        return {"passed": False, "reason": "syntax"}
    with tempfile.TemporaryDirectory(prefix="longctx-grade-") as directory:
        path = Path(directory) / "program.py"
        path.write_text(
            case.get("setup", "") + "\n" + code + "\n" + "\n".join(case["tests"])
        )
        name = "1cat-longctx-grade-" + Path(directory).name
        command = [
            "docker",
            "run",
            "--rm",
            "--runtime",
            "runc",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--user",
            "65534:65534",
            "--memory",
            "512m",
            "--memory-swap",
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--ulimit",
            "cpu=5:5",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "-v",
            f"{path}:/program.py:ro",
            "--entrypoint",
            "/opt/venv/bin/python",
            IMAGE,
            "-I",
            "-S",
            "/program.py",
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            return {"passed": False, "reason": "timeout"}
    return {
        "passed": result.returncode == 0,
        "reason": f"exit_{result.returncode}" if result.returncode else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:18084")
    parser.add_argument("--model", default="quasar-v152-opt")
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    dataset = json.loads(args.cases.read_text())
    digest = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    records = args.output / "results.jsonl"
    identity = {
        "cases_sha256": digest,
        "temperature": 0,
        "max_tokens": 4096,
        "thinking": False,
        "base": args.base,
        "scorer": "final-channel-v1",
        "image_id": args.image_id,
        "model": args.model,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "regex_version": re.__version__,
    }
    identity_path = args.output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("Resume identity differs")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    results = (
        [json.loads(line) for line in records.read_text().splitlines()]
        if records.exists()
        else []
    )
    done = {row["id"] for row in results}
    consecutive_generation_errors = 0
    with records.open("a") as log:
        for case in dataset["cases"]:
            if case["id"] in done:
                continue
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": case["prompt"]}],
                "temperature": 0,
                "seed": 0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                args.base + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            start = time.perf_counter()
            generation_error = False
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                text = choice["message"].get("content") or ""
                evaluation = grade(case, text, choice["finish_reason"])
            except (urllib.error.URLError, TimeoutError) as error:
                timed_out = (
                    isinstance(error, TimeoutError)
                    or isinstance(getattr(error, "reason", None), TimeoutError)
                    or getattr(error, "code", None) in (408, 504)
                )
                evaluation = {
                    "passed": False,
                    "reason": "generation_timeout" if timed_out else "api_transport",
                }
                result = {
                    "benchmark_error": type(error).__name__,
                    "detail": str(error)[:1000],
                }
                generation_error = True
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                evaluation = {"passed": False, "reason": "api_parse"}
                result = {
                    "benchmark_error": type(error).__name__,
                    "detail": str(error)[:1000],
                }
                generation_error = True
            row = {
                "id": case["id"],
                "category": case["category"],
                "response": result,
                "wall_s": time.perf_counter() - start,
                "grade": evaluation,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            results.append(row)
            consecutive_generation_errors = (
                consecutive_generation_errors + 1 if generation_error else 0
            )
            if consecutive_generation_errors >= 3:
                raise RuntimeError(
                    "Three consecutive API failures, failed cases retained"
                )
            if len(results) % 25 == 0:
                print(f"Completed {len(results)}/{len(dataset['cases'])}", flush=True)
    summary = {
        "complete": len(results) == len(dataset["cases"]),
        "total": len(results),
        "categories": {},
    }
    for category in ("mbpp", "gsm8k"):
        rows = [row for row in results if row["category"] == category]
        summary["categories"][category] = {
            "passed": sum(row["grade"]["passed"] for row in rows),
            "total": len(rows),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
