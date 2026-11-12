# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from v152_worker_probe import LongctxWorkerProbe


class QualificationWorker(LongctxWorkerProbe):
    def v152_precision_and_heads(self):
        import torch

        matmul = torch.backends.cuda.matmul
        names = [
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
            "allow_fp16_accumulation",
            "allow_tf32",
        ]
        precision = {
            name: getattr(matmul, name) for name in names if hasattr(matmul, name)
        }
        heads = []
        for name, module in self.model_runner.model.named_modules():
            if not name.endswith("lm_head"):
                continue
            weight = getattr(module, "weight", None)
            state = getattr(module, "_sm70_lm_head_state", None)
            head = {
                "name": name,
                "quant_method": type(getattr(module, "quant_method", None)).__name__,
            }
            if weight is not None:
                head.update(
                    shape=list(weight.shape),
                    dtype=str(weight.dtype),
                    stride=list(weight.stride()),
                )
            if state is not None:
                head["dflash_options"] = {
                    field: getattr(state.dflash, field, None)
                    for field in [
                        "fp32_logits",
                        "qpn8_rerank",
                        "qpn8_rerank_shadow",
                        "target_fp8_qpn8",
                    ]
                }
                head["qpn8_rerank_prepared"] = bool(
                    getattr(state, "_sm70_dflash2_qpn8_rerank_prepared", False)
                )
                head["qpn8_native_buffers"] = {
                    field: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for field in [
                        "_sm70_dflash2_qpn8_codes",
                        "_sm70_dflash2_qpn8_scales",
                        "_sm70_dflash2_qpn8_logits",
                    ]
                    if (value := getattr(state, field, None)) is not None
                }
            heads.append(head)
        return {"rank": self.rank, "matmul_precision": precision, "heads": heads}
