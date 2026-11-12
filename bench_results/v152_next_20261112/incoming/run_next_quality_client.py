# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from pathlib import Path

root = Path("/home/llm/1cat-v152-opt-20261111")
position = sys.argv.index("--grader-image")
grader = sys.argv[position + 1]
assert grader.startswith("sha256:") and len(grader) == 71
int(grader[7:], 16)
del sys.argv[position : position + 2]
sys.path.insert(0, str(root / "benchmarks"))
source = root / "benchmarks/benchmark_v152_admission.py"
spec = importlib.util.spec_from_file_location("v152_fixed_quality_client", source)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.IMAGE = grader
module.main()
