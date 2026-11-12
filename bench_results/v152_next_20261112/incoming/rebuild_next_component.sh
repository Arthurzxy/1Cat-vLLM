#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

if [[ "${1:-}" != "--inside" ]]; then
    test "$#" -eq 3
    bundle_root="$(realpath "$1")"
    source_root="$(realpath "$2")"
    source_commit="$3"
    test "$(git -C "$source_root" rev-parse --show-toplevel)" = "$source_root"
    test "$(git -C "$source_root" rev-parse HEAD)" = "$source_commit"
    test ! -e "$source_root/bench_results/rebuild-next-component/manifest.json"
    docker run --rm --cpus=8 --memory=48g --entrypoint bash \
        -v "$source_root:/workspace" -v "$bundle_root:/artifacts:ro" \
        -w /workspace \
        sha256:8110fe8e0f228f698711703312bfcfb5a40d6063c1be8d734d5ceb7278602ae9 \
        /artifacts/raw/incoming/rebuild_next_component.sh --inside "$source_commit"
    exit
fi

test "$#" -eq 2
source_commit="$2"
git config --global --add safe.directory /workspace
test "$(git rev-parse HEAD)" = "$source_commit"
test -z "$(git diff HEAD --name-only)"
test ! -e /workspace/.venv
/opt/venv/bin/python -m venv --system-site-packages /workspace/.venv
printf '%s\n' /opt/venv/lib/python3.12/site-packages \
    > /workspace/.venv/lib/python3.12/site-packages/base-image.pth
/workspace/.venv/bin/python -m pip install cmake==3.31.6
export PATH="/workspace/.venv/bin:/opt/venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.0

mkdir -p /workspace/bench_results/rebuild-next-deps
cd /workspace/bench_results/rebuild-next-deps
git clone --no-checkout https://github.com/nvidia/cutlass.git cutlass
git -C cutlass checkout --detach da5e086dab31d63815acafdac9a9c5893b1c69e2
git clone --no-checkout https://github.com/zhinianqin/flash-attention-v100.git flash-attn
git -C flash-attn checkout --detach c2eda5e6115b98c3ba4bfd181570668742eece22
git clone --no-checkout https://github.com/triton-lang/triton.git triton
git -C triton checkout --detach 0add68262ab0a2e33b84524346cb27cbb2787356

mkdir -p /workspace/bench_results/rebuild-next-component/build
cat > /workspace/bench_results/rebuild-next-component/build/CMakeCache.txt <<'CACHE'
CMAKE_CUDA_ARCHITECTURES:STRING=70
FETCHCONTENT_SOURCE_DIR_CUTLASS:PATH=/workspace/bench_results/rebuild-next-deps/cutlass
FETCHCONTENT_SOURCE_DIR_VLLM-FLASH-ATTN:PATH=/workspace/bench_results/rebuild-next-deps/flash-attn
CACHE
cd /workspace
/workspace/.venv/bin/python \
    /artifacts/raw/incoming/verify_next_component_inputs.py \
    --source /workspace --expected-commit "$source_commit" \
    --inputs /artifacts/raw/native-lt-component/inputs.json
/workspace/.venv/bin/python benchmarks/build_v152_component.py \
    --source-root /workspace --expected-source-commit "$source_commit" \
    --run-dir /workspace/bench_results/rebuild-next-component \
    --triton-source /workspace/bench_results/rebuild-next-deps/triton \
    --target _C --jobs 4
printf '%s\n' 'Component output is in bench_results/rebuild-next-component/install/vllm.'
