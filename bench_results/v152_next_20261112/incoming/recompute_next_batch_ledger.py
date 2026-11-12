# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
out = parser.parse_args().root
folder = out / "batch-ledger"
assert json.loads((folder / "admission.json").read_text())["complete"]
report = {
    "complete": True,
    "cases": [],
    "scope": (
        "Real default async service. Unique worker batch IDs and nonoverlapping "
        "main-stream event intervals are counted once. Exact returned token IDs "
        "assign API-visible commits to sampling batches. Instrumentation fences and"
        " CPU build concurrency prevent formal speed interpretation."
    ),
}
for name in ["c1-tail33", "c2-mixed-tail33", "natural-eos"]:
    path = folder / (name + ".json")
    data = json.loads(path.read_text())
    assert data["complete"]
    rows = data["ledger"]["results"][0]["rows"]
    requests = [item for row in rows for item in row["requests"]]
    for response in data["responses"]:
        matching = [
            item for item in requests if item["request_id"].startswith(response["id"])
        ]
        raw_tokens = [
            token for item in matching for token in item["raw_sampled_token_ids"]
        ]
        assert raw_tokens[: len(response["tokens"])] == response["tokens"]
        assert sum(item["api_committed_tokens"] for item in matching) == len(
            response["tokens"]
        )
    assert all("api_committed_tokens" in item for item in requests)
    pure = [
        row
        for row in rows
        if row["requests"] and all(item["strict_decode"] for item in row["requests"])
    ]
    mixed = [
        row
        for row in rows
        if any(item["strict_decode"] for item in row["requests"])
        and any(not item["strict_decode"] for item in row["requests"])
    ]
    row = {
        "case": name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unique_engine_batches": len(rows),
        "raw_post_rejection_gpu_tokens": sum(
            item["raw_num_sampled"] for item in requests
        ),
        "api_committed_tokens": sum(item["api_committed_tokens"] for item in requests),
        "api_excluded_raw_tail": sum(
            item["api_excluded_raw_tokens"] for item in requests
        ),
        "prefill_first_or_other_non_decode_commits": sum(
            item["api_committed_tokens"]
            for item in requests
            if not item["strict_decode"]
        ),
        "strict_decode_api_commits": sum(
            item["api_committed_tokens"] for item in requests if item["strict_decode"]
        ),
        "raw_proposed_tokens": sum(item["proposal_tokens"] for item in requests),
        "raw_accepted_proposals": sum(item["accepted_proposals"] for item in requests),
        "visible_accepted_proposals": sum(
            item["visible_accepted_proposals"] for item in requests
        ),
        "visible_all_accepted_bonus": sum(item["visible_bonus"] for item in requests),
        "visible_rejection_correction": sum(
            item["visible_rejection_correction"] for item in requests
        ),
        "non_spec_gpu_samples": sum(item["non_spec_sample"] for item in requests),
        "pure_decode_unique_batches": len(pure),
        "pure_decode_batch_api_commits": sum(
            item["api_committed_tokens"] for step in pure for item in step["requests"]
        ),
        "pure_decode_batch_instrumented_ms": sum(step["gpu_batch_ms"] for step in pure),
        "mixed_prefill_decode_batches": len(mixed),
        "mixed_batch_decode_api_commits": sum(
            item["api_committed_tokens"]
            for step in mixed
            for item in step["requests"]
            if item["strict_decode"]
        ),
        "commits_per_unique_batch": [
            sum(item["api_committed_tokens"] for item in step["requests"])
            for step in rows
        ],
        "stop_reasons": [response["finish_reason"] for response in data["responses"]],
        "requested_output_budgets": [
            response["request"]["max_tokens"] for response in data["responses"]
        ],
        "actual_output_tokens": [
            len(response["tokens"]) for response in data["responses"]
        ],
        "nonoverlap_verified": all(
            rows[index]["gpu_start_ms"] >= rows[index - 1]["gpu_end_ms"] - 0.001
            for index in range(1, len(rows))
        ),
    }
    assert (
        row["raw_post_rejection_gpu_tokens"]
        == row["api_committed_tokens"] + row["api_excluded_raw_tail"]
    )
    assert (
        row["strict_decode_api_commits"]
        == row["pure_decode_batch_api_commits"] + row["mixed_batch_decode_api_commits"]
    )
    assert row["api_committed_tokens"] == sum(row["actual_output_tokens"])
    if name == "natural-eos":
        assert row["stop_reasons"] == ["stop"]
        assert row["actual_output_tokens"][0] < row["requested_output_budgets"][0]
    report["cases"].append(row)
(out / "BATCH_LEDGER_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        [
            {
                key: value
                for key, value in row.items()
                if key != "commits_per_unique_batch"
            }
            for row in report["cases"]
        ],
        indent=2,
    )
)
