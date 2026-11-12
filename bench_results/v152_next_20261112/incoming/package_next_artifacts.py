# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a fully indexed cumulative archive with normalized PAX paths."""

import copy
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

root = Path("/home/llm/1cat-v152-opt-20261111")
out = root / "bench_results/v152_next_20261112"
delivery = root / "delivery-v152-next-20261112"
delivery.mkdir(exist_ok=True)
for name in ["FINAL_QUALITY_SUMMARY.json", "FINAL_QUALIFICATION.json", "CLEANUP.json"]:
    assert json.loads((out / name).read_text())["complete"], name
previous = root / "bench_results/v152_opt_20261111/delivery/v152-opt-artifacts.tar.gz"
old_hash = "28427ba260a458bf4185d096ce3b60cff67b2c1f28b6a5f6f928f79887b73ce3"
assert hashlib.file_digest(previous.open("rb"), "sha256").hexdigest() == old_hash
archive = delivery / "v152-next-artifacts.tar.gz"
assert not archive.exists()
prefix = "v152-next-artifacts/"
index = {}
repaired = []
excluded = []


def safe_name(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts and name
    return str(path)


def add(bundle, info, data=None):
    name = safe_name(info.name)
    assert name not in index, name
    info = copy.copy(info)
    info.name = prefix + name
    info.pax_headers = {
        key: value
        for key, value in info.pax_headers.items()
        if key not in ["path", "linkpath"]
    }
    if info.isdir():
        return
    assert info.isfile() or info.issym(), (name, info.type)
    if info.issym():
        assert not PurePosixPath(info.linkname).is_absolute()
        index[name] = {"kind": "symlink", "target": info.linkname, "mode": info.mode}
        bundle.addfile(info)
        return
    assert data is not None and len(data) == info.size
    index[name] = {
        "kind": "file",
        "bytes": len(data),
        "mode": info.mode,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    bundle.addfile(info, io.BytesIO(data))


with tarfile.open(archive, "w:gz", compresslevel=6) as bundle:
    with tarfile.open(previous, "r|gz") as source:
        for info in source:
            if info.name.startswith("v152-opt-artifacts/"):
                name = "previous/" + info.name.removeprefix("v152-opt-artifacts/")
            else:
                assert info.name.startswith("vllm/") and info.name.endswith(".json"), (
                    info.name
                )
                name = "previous/source-b0/" + info.name
                repaired.append({"old_name": info.name, "new_name": name})
            data = source.extractfile(info).read() if info.isfile() else None
            updated = copy.copy(info)
            updated.name = name
            add(bundle, updated, data)
    assert len(repaired) == 3
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if any(
            part in ["__pycache__", ".pytest_cache", ".ruff_cache"]
            for part in relative.parts
        ):
            continue
        if relative.parts[0] == "bsafe-service-cache" or relative.parts[:2] == (
            "native-lt-component",
            "build",
        ):
            if path.is_file():
                excluded.append(str(relative))
            continue
        if path.is_dir():
            continue
        assert not path.is_symlink(), path
        info = bundle.gettarinfo(str(path), arcname="raw/" + str(relative))
        add(bundle, info, path.read_bytes())
    process = subprocess.Popen(
        ["git", "archive", "HEAD"], cwd=root, stdout=subprocess.PIPE
    )
    with tarfile.open(fileobj=process.stdout, mode="r|") as source:
        for info in source:
            data = source.extractfile(info).read() if info.isfile() else None
            updated = copy.copy(info)
            updated.name = "source-final/" + info.name
            add(bundle, updated, data)
    assert process.wait() == 0
    for name in [
        "FINAL_REPORT.txt",
        "RESUME.txt",
        "REPRODUCE.txt",
        "BUILD_MANIFEST.json",
        "FINAL_QUALIFICATION.json",
        "EXPERIMENTS.json",
        "UPSTREAM_LEDGER.json",
        "VALIDATION_SUMMARY.json",
        "retained-source.patch",
        "next-runtime.patch",
        "Dockerfile.replay-bprev",
        "Dockerfile.replay-bsafe",
        "Dockerfile.replay-bnew",
        ".dockerignore",
    ]:
        path = out / name
        add(bundle, bundle.gettarinfo(str(path), arcname=name), path.read_bytes())
    metadata = {
        "complete": True,
        "previous_archive_sha256": old_hash,
        "repaired_pax_paths": repaired,
        "source_final_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_snapshot_role": (
            "Content reference. Native rebuilding requires an exact Git checkout "
            "and recorded dependencies."
        ),
        "omitted_regenerable_files": excluded,
        "inventory_scope": (
            "All delivered regular files and symlinks, including Dockerfiles, "
            "source snapshots, scripts, licenses and executable modes. FILES.json "
            "is the sole self-index exception. External SHA256SUMS covers the "
            "archive."
        ),
    }
    data = (json.dumps(metadata, indent=2) + "\n").encode()
    info = tarfile.TarInfo("PACKAGING.json")
    info.size = len(data)
    info.mode = 0o644
    add(bundle, info, data)
    index_data = (json.dumps(index, indent=2) + "\n").encode()
    info = tarfile.TarInfo(prefix + "FILES.json")
    info.size = len(index_data)
    info.mode = 0o644
    bundle.addfile(info, io.BytesIO(index_data))
(delivery / "FILES.json").write_bytes(index_data)
digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
(delivery / "SHA256SUMS.txt").write_text(digest + "  " + archive.name + "\n")
(delivery / "PACKAGING.json").write_text(
    json.dumps(
        {
            **metadata,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": digest,
            "indexed_entries": len(index),
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {
            "archive": str(archive),
            "sha256": digest,
            "bytes": archive.stat().st_size,
            "indexed_entries": len(index),
            "pax_repairs": len(repaired),
        }
    ),
    flush=True,
)
