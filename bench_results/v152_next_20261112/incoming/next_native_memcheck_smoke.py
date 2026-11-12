# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Six retained projection shapes through the shipped native entrypoint."""

import hashlib
import json
import sys
from pathlib import Path

import torch

from vllm import _C, _sm70_ops

sys.path.insert(0, "/workspace")
from benchmarks.kernels import benchmark_sm70_quasar_nvfp4_oracle as oracle

root = Path("/workspace")
out = root / "bench_results/v152_next_20261112"

projections = {
    projection.name: projection
    for projection in oracle._load_projections(
        Path("/data/models/Qwen3.8-27B-QUASAR-NVFP4"), 0, 4
    )
}
baseline = json.loads((out / "native-dso-unit/baseline.json").read_text())
rows = []
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
for path in sorted((out / "real-activation-capture/projections").glob("*.pt")):
    captured = torch.load(path, weights_only=False)
    layer = captured["layer_metadata"]
    state = captured["state_metadata"]
    original = captured["input"]
    n = layer["sm70_nvfp4_qpn2_output_size"]
    if original.shape[1] != 5120 or n not in [3584, 4128, 8704]:
        continue
    projection = {3584: "attention_qkv", 4128: "gdn_qkvzba", 8704: "mlp_gate_up"}[n]
    weight = captured["state_tensors"]["weight"].cuda()
    scales = captured["state_tensors"]["scales"].cuda()
    raw_scales = _sm70_ops.nvfp4_qpn2_prepare_scales_sm70(
        projections[projection].scales.cuda()
    )
    for m in [4096, 8192]:
        x = original[:m].cuda().contiguous()
        output = torch.empty(
            m, n // 2 if captured["gated"] else n, device="cuda", dtype=torch.float16
        )

        def launch(
            output=output,
            x=x,
            weight=weight,
            raw_scales=raw_scales,
            layer=layer,
            scales=scales,
            state=state,
            captured=captured,
        ):
            torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
                output,
                x,
                weight,
                raw_scales,
                layer["sm70_nvfp4_qpn2_global_scale"],
                layer["sm70_nvfp4_qpn2_split_k"],
                layer["sm70_nvfp4_qpn2_nacc"],
                scales,
                state["group_size"],
                state["k_ld"],
                state["q_ld"],
                captured["gated"],
                layer["sm70_nvfp4_qpn2_prefill_min_m"],
                state["prescaled_scales"],
            )

        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph.replay()
        torch.accelerator.synchronize()
        digest = hashlib.sha256(
            output.cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        key = f"{projection}-m{m}-scale1.0"
        expected = next(row for row in baseline["rows"] if row["key"] == key)
        assert digest == expected["output_sha256"]
        rows.append(
            {
                "key": key,
                "baseline_byte_equal": True,
                "output_sha256": digest,
                "graph_replay": True,
            }
        )
        del graph, output, x
    del weight, scales, raw_scales
assert len(rows) == 6
(out / "NATIVE_MEMCHECK_SMOKE.json").write_text(
    json.dumps(
        {
            "complete": True,
            "rows": rows,
            "library_sha256": hashlib.sha256(
                Path(_C.__file__).read_bytes()
            ).hexdigest(),
            "scope": (
                "Six selected real projection shapes, eager/capture/replay under "
                "external compute-sanitizer memcheck. Sanitizer log is required to "
                "establish memory-check success."
            ),
        },
        indent=2,
    )
    + "\n"
)
