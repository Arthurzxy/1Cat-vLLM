# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify archive prefix, all indexed bytes, executable modes and symlinks."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
parser.add_argument("--sha256", required=True)
args = parser.parse_args()
assert hashlib.file_digest(args.archive.open("rb"), "sha256").hexdigest() == args.sha256
prefix = "v152-next-artifacts/"
actual = {}
index = None
with tarfile.open(args.archive, "r|gz") as bundle:
    for info in bundle:
        assert info.name.startswith(prefix), info.name
        name = info.name.removeprefix(prefix)
        assert ".." not in PurePosixPath(name).parts
        assert name not in actual
        if info.isdir():
            continue
        if name == "FILES.json":
            index = json.load(bundle.extractfile(info))
            continue
        if info.issym():
            actual[name] = {
                "kind": "symlink",
                "target": info.linkname,
                "mode": info.mode,
            }
        else:
            assert info.isfile(), (name, info.type)
            stream = bundle.extractfile(info)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            actual[name] = {
                "kind": "file",
                "bytes": info.size,
                "mode": info.mode,
                "sha256": digest,
            }
assert index == actual, "Delivered members and inventory differ"
assert any(name.startswith("previous/source-b0/") for name in index)
assert any(name.startswith("source-final/") for name in index)
assert "previous/Dockerfile.bbest" in index
assert any(
    "Blackwell_Server_Edition" in name and name.startswith("previous/source-b0/")
    for name in index
)
print(
    json.dumps(
        {
            "complete": True,
            "entries_verified": len(index),
            "archive_sha256": args.sha256,
            "prefix_verified": prefix,
            "modes_verified": True,
        }
    ),
    flush=True,
)
