# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate cumulative patches and verify their exact Git tree effects."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
assert json.loads((out / "FINAL_REPORTS_READY.json").read_text())["complete"]
assert not subprocess.check_output(
    ["git", "diff", "HEAD", "--name-only"], cwd=root, text=True
).strip()
head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()
base = "2bb8c2b9cd7c1696630b6be464052027f0dd0147"
previous = "0a9ca74bc4897b8a8c3a2b9cbf7c345a77d063b9"
records = {}
for filename, anchor, paths in [
    ("retained-source.patch", base, []),
    ("next-runtime.patch", previous, ["csrc", "vllm"]),
]:
    command = ["git", "diff", "--binary", anchor, head]
    if paths:
        command += ["--", *paths]
    patch = subprocess.check_output(command, cwd=root)
    path = out / filename
    path.write_bytes(patch)
    index = out / (filename + ".verification-index")
    assert not index.exists()
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index)
    try:
        subprocess.run(
            ["git", "read-tree", anchor], cwd=root, env=environment, check=True
        )
        subprocess.run(
            ["git", "apply", "--cached", "--check", str(path)],
            cwd=root,
            env=environment,
            check=True,
        )
        subprocess.run(
            ["git", "apply", "--cached", str(path)],
            cwd=root,
            env=environment,
            check=True,
        )
        tree = subprocess.check_output(
            ["git", "write-tree"], cwd=root, env=environment, text=True
        ).strip()
        difference = [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            tree,
            head,
        ]
        if paths:
            difference += ["--", *paths]
        assert not subprocess.check_output(difference, cwd=root, text=True).strip()
    finally:
        index.unlink(missing_ok=True)
    records[filename] = {
        "anchor": anchor,
        "source_final_commit": head,
        "paths": paths or ["all tracked task files"],
        "sha256": hashlib.sha256(patch).hexdigest(),
        "bytes": len(patch),
        "temporary_index_applied_tree": tree,
        "exact_tree_effect_verified_in_scope": True,
    }
(out / "PATCH_APPLICABILITY_QA.json").write_text(
    json.dumps(
        {
            "complete": True,
            "patches": records,
            "scope": (
                "Independent temporary Git indexes. The checkout and normal index "
                "remain unchanged."
            ),
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(records, indent=2), flush=True)
