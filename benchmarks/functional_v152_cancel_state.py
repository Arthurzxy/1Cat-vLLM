# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cancel after real decode, observe physical reuse and compare clean state."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", default="http://127.0.0.1:18111")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tokens-file", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
fixture = (
    args.tokens_file
    or root / "bench_results/v152_opt_20261111/unique-corpus-tokens.json"
)
corpus = json.loads(fixture.read_text())["tokens"]
assert len(corpus) > 4096
assert not args.output.exists()
args.output.parent.mkdir(parents=True, exist_ok=True)
report = {
    "complete": False,
    "passed": False,
    "input_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
}


def save():
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def post(path, payload):
    request = urllib.request.Request(
        args.base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = response.read()
        return json.loads(body) if body else None


def rpc(method, **kwargs):
    return post(
        "/collective_rpc", {"method": method, "kwargs": kwargs, "timeout": 120}
    )["results"]


def wait_idle():
    deadline = time.monotonic() + 30
    while True:
        with urllib.request.urlopen(args.base + "/metrics", timeout=10) as response:
            metrics = response.read().decode()
        active = [
            float(line.rsplit(" ", 1)[1])
            for line in metrics.splitlines()
            if line.startswith(
                ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
            )
        ]
        assert active, "Missing running/waiting metrics"
        if not any(active):
            return metrics
        assert time.monotonic() < deadline, (
            "Request remains resident after cancellation"
        )
        time.sleep(0.1)


def has_decode_state(values):
    return len(values) == 4 and all(
        row["active_snapshot"] is not None
        and row["active_snapshot"]["computed_tokens_gpu"]
        > row["active_snapshot"]["prompt_tokens"]
        and row["active_snapshot"]
        .get("gdn_state", {})
        .get("tensors", {})
        .get("ssm", {})
        .get("nonzero", 0)
        > 0
        for row in values
    )


def complete(prompt, seed, budget=64, cancel=False):
    payload = {
        "model": "quasar-v152-opt",
        "prompt": prompt,
        "max_tokens": budget,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.95,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        args.base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    value = {
        "id": None,
        "tokens": [],
        "nonempty_text_chunks": 0,
        "usage": None,
        "cancelled": False,
    }
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            assert not event.get("error"), event
            value["id"] = event.get("id", value["id"])
            value["usage"] = event.get("usage") or value["usage"]
            for choice in event.get("choices", []):
                value["tokens"].extend(choice.get("token_ids") or [])
                value["nonempty_text_chunks"] += bool(choice.get("text"))
            if cancel and len(value["tokens"]) >= 8 and value["nonempty_text_chunks"]:
                snapshots = rpc("v152_request_state_trace", request_id=value["id"])
                if has_decode_state(snapshots):
                    value["before_cancel"] = snapshots
                    value["cancelled"] = True
                    break
    if cancel:
        assert value["cancelled"], (
            "No real decode/state-write point observed before close"
        )
    else:
        assert len(value["tokens"]) == budget
        assert value["usage"]["completion_tokens"] == budget
    return value


def final_snapshot(row, public_id):
    if row["active_snapshot"] is not None:
        return row["active_snapshot"]
    matching = [
        event["snapshot"]
        for event in row["events"]
        if event["event"] == "remove" and event["request_id"].startswith(public_id)
    ]
    assert len(matching) == 1
    return matching[0]


save()
report["trace_enabled"] = rpc("v152_begin_request_state_trace")
reference_prompt = corpus[1024:1280]
post("/reset_prefix_cache", {})
report["reference"] = complete(reference_prompt, 173)
report["reference_state"] = rpc(
    "v152_request_state_trace", request_id=report["reference"]["id"]
)
save()
report["cancel"] = complete(corpus[2048:3072], 991, budget=4096, cancel=True)
report["after_cancel_idle_metrics"] = wait_idle()
save()
post("/reset_prefix_cache", {})
report["followup"] = complete(reference_prompt, 173)
report["followup_state"] = rpc(
    "v152_request_state_trace", request_id=report["followup"]["id"]
)
save()
assert report["reference"]["tokens"] == report["followup"]["tokens"]
report["flush"] = complete(corpus[3072:3328], 177, budget=1)
report["events"] = rpc("v152_request_state_trace")
save()
checks = []
for index, trace in enumerate(report["events"]):
    old = report["cancel"]["before_cancel"][index]["active_snapshot"]
    reference = final_snapshot(
        report["reference_state"][index], report["reference"]["id"]
    )
    followup = final_snapshot(report["followup_state"][index], report["followup"]["id"])
    removed = [
        event
        for event in trace["events"]
        if event["event"] == "remove" and event["request_id"] == old["request_id"]
    ]
    added = [
        event
        for event in trace["events"]
        if event["event"] == "add"
        and event["request_id"].startswith(report["followup"]["id"])
    ]
    assert len(removed) == len(added) == 1
    assert removed[0]["sequence"] < added[0]["sequence"]
    assert added[0]["slot"] == old["slot"] == followup["slot"]
    assert added[0]["initial_computed_tokens"] == 0 and added[0]["draft_zero"]
    reset = next(
        event
        for event in trace["events"]
        if event["event"] == "model_add" and event["sequence"] > added[0]["sequence"]
    )
    assert reset["accepted_neutral"] == 1
    assert reset["align_state_index"] in (None, -2)
    assert reference["computed_tokens_gpu"] == followup["computed_tokens_gpu"]
    for name in ["conv", "ssm"]:
        a, b = [row["gdn_state"]["tensors"][name] for row in (reference, followup)]
        assert a["nonzero"] > 0 and b["nonzero"] > 0 and a["finite"] and b["finite"]
        assert a["sha256"] == b["sha256"], (index, name, a, b)
    checks.append(
        {
            "rank": trace["rank"],
            "reused_slot": old["slot"],
            "old_removed_before_new_add": True,
            "state_hashes_equal": True,
        }
    )
report.update(
    {
        "complete": True,
        "passed": True,
        "rank_checks": checks,
        "followup_tokens_match": True,
    }
)
save()
print(json.dumps({"passed": True, "rank_checks": checks}))
