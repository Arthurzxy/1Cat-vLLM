# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify an exact Git checkout against the recorded native source inputs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--expected-commit", required=True)
args = parser.parse_args()
source = args.source.resolve()
actual_root = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True
).strip()
assert Path(actual_root).resolve() == source
commit = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip()
assert commit == args.expected_commit
inputs = json.loads(args.inputs.read_text())
for name, digest in inputs.items():
    actual = hashlib.file_digest((source / name).open("rb"), "sha256").hexdigest()
    assert actual == digest, name
print(
    json.dumps(
        {
            "complete": True,
            "source_commit": commit,
            "matched_native_inputs": len(inputs),
        }
    )
)
