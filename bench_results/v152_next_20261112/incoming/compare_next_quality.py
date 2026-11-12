# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute final quality gates from all frozen case records and NLL rows."""

import argparse
import gzip
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
out = args.root
runs = json.loads((out / "QUALITY_SERVICE_RUNS_COMPLETE.json").read_text())
gates = runs["gates"]
arms = {}
for run in runs["arms"]:
    arm = run["arm"]
    folder = out / ("quality-" + arm)
    records = [
        json.loads(line)
        for line in (folder / "mbpp-gsm8k/results.jsonl").read_text().splitlines()
    ]
    assert len(records) == len({row["id"] for row in records}) == 1819
    assert all("benchmark_error" not in row["response"] for row in records)
    functional = json.loads((folder / "functional.json").read_text())
    assert functional["complete"] and len(functional["checks"]) == 50
    cancel = json.loads((folder / "cancel-state.json").read_text())
    assert cancel["complete"] and cancel["passed"]
    ppl = json.loads((folder / "ppl.json").read_text())
    assert ppl["complete"] and len(ppl["cases"]) == 296
    scores = {
        category: {
            "passed": sum(
                row["grade"]["passed"] for row in records if row["category"] == category
            ),
            "total": sum(row["category"] == category for row in records),
        }
        for category in ["mbpp", "gsm8k"]
    }
    assert scores["mbpp"]["total"] == 500 and scores["gsm8k"]["total"] == 1319
    arms[arm] = {
        "scores": scores,
        "ppl": ppl["ppl"],
        "nll_cases": ppl["cases"],
        "functional_passed": sum(check["passed"] for check in functional["checks"]),
        "functional_failures": [
            check["label"] for check in functional["checks"] if not check["passed"]
        ],
        "cancel_state_passed": True,
        "image_id": run["image_id"],
    }
comparison = None
passed = True
if "new" in arms:
    safe = arms["safe"]
    new = arms["new"]
    safe_records = {
        row["id"]: row
        for row in map(
            json.loads,
            (out / "quality-safe/mbpp-gsm8k/results.jsonl").read_text().splitlines(),
        )
    }
    new_records = {
        row["id"]: row
        for row in map(
            json.loads,
            (out / "quality-new/mbpp-gsm8k/results.jsonl").read_text().splitlines(),
        )
    }
    assert safe_records.keys() == new_records.keys()
    message_changes = [
        key
        for key in safe_records
        if safe_records[key]["response"]["choices"][0]["message"]
        != new_records[key]["response"]["choices"][0]["message"]
    ]
    grade_changes = [
        key
        for key in safe_records
        if safe_records[key]["grade"] != new_records[key]["grade"]
    ]
    relative_ppl = new["ppl"] / safe["ppl"] - 1
    nll_deltas = [
        abs(a["nll_sum"] / a["count"] - b["nll_sum"] / b["count"])
        for a, b in zip(safe["nll_cases"], new["nll_cases"])
    ]
    for a, b in zip(safe["nll_cases"], new["nll_cases"]):
        assert {key: value for key, value in a.items() if key != "nll_sum"} == {
            key: value for key, value in b.items() if key != "nll_sum"
        }
    safe_functional = json.loads((out / "quality-safe/functional.json").read_text())[
        "checks"
    ]
    new_functional = json.loads((out / "quality-new/functional.json").read_text())[
        "checks"
    ]
    verdict_changes = [
        a["label"]
        for a, b in zip(safe_functional, new_functional)
        if a["label"] != b["label"] or a["passed"] != b["passed"]
    ]
    category_pass = all(
        new["scores"][category]["passed"] >= safe["scores"][category]["passed"]
        for category in ["mbpp", "gsm8k"]
    )
    logprob_count = 0
    max_logprob_delta = 0.0
    for path in sorted((out / "quality-safe/ppl.rows").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            left = json.load(stream)
        with gzip.open(out / "quality-new/ppl.rows" / path.name, "rt") as stream:
            right = json.load(stream)
        assert left["tokens"] == right["tokens"]
        for token, a, b in zip(
            left["tokens"], left["prompt_logprobs"], right["prompt_logprobs"]
        ):
            if a is None:
                assert b is None
                continue
            delta = abs(a[str(token)]["logprob"] - b[str(token)]["logprob"])
            max_logprob_delta = max(max_logprob_delta, delta)
            logprob_count += 1
    passed = category_pass and abs(relative_ppl) <= gates["ppl_relative_change_max"]
    passed = (
        passed
        and max(nll_deltas) <= gates["nll_absolute_change_max"]
        and not verdict_changes
    )
    comparison = {
        "category_gate_passed": category_pass,
        "ppl_relative_change": relative_ppl,
        "max_per_case_mean_nll_abs_change": max(nll_deltas),
        "message_changed_case_ids": message_changes,
        "actual_token_logprobs_compared": logprob_count,
        "max_actual_token_logprob_abs_change": max_logprob_delta,
        "grade_changed_case_ids": grade_changes,
        "functional_verdict_changed_labels": verdict_changes,
        "all_messages_identical": not message_changes,
        "all_grades_identical": not grade_changes,
    }
report = {
    "complete": True,
    "passed": passed,
    "arms": arms,
    "comparison": comparison,
    "gates": gates,
    "limits": (
        "Known marker omissions remain visible in all50 functional verdicts. Paired"
        " agreement does not establish unquantized equivalence or absolute model "
        "correctness."
    ),
}
(out / "FINAL_QUALITY_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "passed": passed,
            "comparison": comparison,
            "scores": {arm: value["scores"] for arm, value in arms.items()},
        }
    ),
    flush=True,
)
assert passed, "Preserve raw evidence and withdraw candidate before final qualification"
