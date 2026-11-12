# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual CLI token accounting and pre-build source validation."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("V152_TEST_REPO_ROOT", Path(__file__).resolve().parents[2]))


@pytest.mark.parametrize("natural_eos", [False, True])
@pytest.mark.parametrize("concurrency", [1, 2])
def test_streaming_cli_uses_actual_emitted_tokens(tmp_path, natural_eos, concurrency):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
            )

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            if self.path == "/tokenize":
                self.wfile.write(json.dumps({"tokens": [65, 66, 67]}).encode())
            elif self.path == "/v1/completions":
                tokens = [100, 101, 102]
                event = {
                    "choices": [
                        {"text": "abc", "token_ids": tokens, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": len(payload["prompt"]),
                        "completion_tokens": len(tokens),
                    },
                }
                self.wfile.write(
                    ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
                )
            else:
                assert self.path == "/reset_prefix_cache"
                self.wfile.write(b"{}")

    binary = tmp_path / "nvidia-smi"
    binary.write_text(
        "#!" + sys.executable + "\nprint('6, 0 MiB, 0 %, 35, 1290 MHz, 54 W')\n"
    )
    binary.chmod(0o755)
    fixture = tmp_path / "input.json"
    fixture.write_text(json.dumps({"tokens": list(range(128))}))
    output = tmp_path / "result.json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    arguments = [
        sys.executable,
        str(ROOT / "benchmarks/benchmark_v152.py"),
        "--base",
        "http://127.0.0.1:" + str(server.server_port),
        "--output",
        str(output),
        "--tokens-file",
        str(fixture),
        "--lengths",
        "128",
        "--repeats",
        "1",
        "--concurrency",
        str(concurrency),
        "--output-tokens",
        "16" if natural_eos else "3",
    ]
    if natural_eos:
        arguments.append("--natural-eos")
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    try:
        subprocess.run(arguments, env=env, check=True, capture_output=True, text=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    result = json.loads(output.read_text())
    assert result["complete"]
    row = result["runs"][0]
    assert row["emitted_tokens"] == concurrency * 3
    assert all(len(item["tokens"]) == 3 for item in row["results"])
    assert row["aggregate_output_tps"] * row["batch_wall_s"] == pytest.approx(
        concurrency * 3
    )


@pytest.mark.parametrize("nested_in_git", [False, True])
def test_archived_source_is_rejected_before_build(tmp_path, nested_in_git):
    if nested_in_git:
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    source = tmp_path / "source-b0"
    source.mkdir()
    destination = tmp_path / "build-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/build_v152_component.py"),
            "--source-root",
            str(source),
            "--run-dir",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "exact top-level Git checkout" in result.stderr
    assert not destination.exists()


def test_wrong_expected_commit_is_rejected_before_build(tmp_path):
    source = tmp_path / "checkout"
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    destination = tmp_path / "build-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/build_v152_component.py"),
            "--source-root",
            str(source),
            "--expected-source-commit",
            "0" * 40,
            "--run-dir",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "differs from expected" in result.stderr
    assert not destination.exists()


@pytest.mark.parametrize(
    "event, expected",
    [
        ({"choices": [{"delta": {"role": "assistant"}}]}, False),
        ({"choices": [{"delta": {"content": ""}}]}, False),
        ({"choices": [{"delta": {"content": None}}]}, False),
        ({"choices": [{"delta": {"content": "token"}}]}, True),
        ({"choices": [{"delta": {"reasoning": "thinking"}}]}, True),
        ({"choices": [{"delta": {"reasoning_content": "thinking"}}]}, True),
        ({"choices": [{"text": "generated"}]}, True),
        ({"choices": [], "usage": {"completion_tokens": 1}}, False),
    ],
)
def test_generated_text_excludes_role_and_empty_events(event, expected):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "v152_validation", ROOT / "benchmarks/v152_validation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.has_generated_text(event) is expected
