# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only rank-local identity and route counters for localhost benchmarks."""


class LongctxWorkerProbe:
    def longctx_identity(self):
        import hashlib
        import sys
        from pathlib import Path

        import torch

        from vllm.config.execution_policy import flash_v100_policy
        from vllm.diagnostics import diagnostics_for
        from vllm.v1.attention.backends.flash_v100 import routing

        module = sys.modules.get("vllm.vllm_flash_attn._vllm_fa2_C")
        paths = {p for p in torch.ops.loaded_libraries if "_vllm_fa2_C" in p}
        if module is not None and getattr(module, "__file__", None):
            paths.add(module.__file__)
        core = sys.modules.get("vllm._C")
        core_paths = {core.__file__} if core is not None else set()
        owner = diagnostics_for(self.vllm_config)
        histories = owner.histories if owner is not None else {}
        cache = self.model_runner.kv_cache_config
        groups = [
            {
                "type": type(group.kv_cache_spec).__name__,
                "block_size": group.kv_cache_spec.block_size,
                "sliding_window": getattr(group.kv_cache_spec, "sliding_window", None),
                "layers": len(group.layer_names),
                "dtypes": [
                    str(dtype) for dtype in getattr(group.kv_cache_spec, "dtypes", ())
                ],
            }
            for group in cache.kv_cache_groups
        ]
        spec = self.vllm_config.speculative_config
        return {
            "dflash_options": spec.sm70_dflash2.graph_options() if spec else None,
            "rank": self.rank,
            "cache_blocks": cache.num_blocks,
            "cache_groups": groups,
            "device": str(self.device),
            "prefill_policy": list(
                flash_v100_policy(self.vllm_config).options.prefill_native_effective
            ),
            "libraries": {
                str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in sorted(paths)
            },
            "core_libraries": {
                str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in sorted(core_paths)
            },
            "routes": dict(histories.get("flash_v100_routes", routing._route_counts)),
            "fallbacks": dict(
                histories.get("flash_v100_fallbacks", routing._fallback_counts)
            ),
        }

    def longctx_projection_shapes(self):
        shapes = {}
        for name, module in self.model_runner.model.named_modules():
            if not hasattr(module, "input_size_per_partition") or not hasattr(
                module, "output_size_per_partition"
            ):
                continue
            key = (
                int(module.input_size_per_partition),
                int(module.output_size_per_partition),
            )
            item = shapes.setdefault(
                key, {"k": key[0], "n": key[1], "count": 0, "examples": []}
            )
            state = getattr(module, "_sm70_turbomind_linear", None)
            if state is not None:
                item["quantized_state"] = {
                    "op_kind": state.op_kind,
                    "gated_silu": bool(
                        getattr(module, "sm70_nvfp4_qpn2_gated_silu", state.gated_silu)
                    ),
                    "qpn2_native": bool(
                        getattr(module, "sm70_nvfp4_qpn2_native", False)
                    ),
                    "codes_contiguous": state.weight.is_contiguous(),
                    "split_k": state.split_k,
                    "accumulator_chains": state.accumulator_chains,
                    "use_scale_code": state.use_scale_code,
                    "global_scale": state.global_scale,
                }
            providers = item.setdefault("provider_counts", {})
            kind = getattr(state, "op_kind", "unquantized")
            providers[kind] = providers.get(kind, 0) + 1
            native_counts = item.setdefault("native_counts", {})
            native = str(bool(getattr(module, "sm70_nvfp4_qpn2_native", False)))
            native_counts[native] = native_counts.get(native, 0) + 1
            item["count"] += 1
            if len(item["examples"]) < 3:
                item["examples"].append(name)
        return {"rank": self.rank, "shapes": list(shapes.values())}

    def longctx_memory(self, reset=False):
        import torch

        if reset:
            torch.accelerator.reset_peak_memory_stats(self.device)
        free, total = torch.accelerator.get_memory_info(self.device)
        return {
            "rank": self.rank,
            "allocated_bytes": torch.accelerator.memory_allocated(self.device),
            "reserved_bytes": torch.accelerator.memory_reserved(self.device),
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(self.device),
            "observed_device_used_bytes": total - free,
        }

    def v152_graph_trace(self, enabled=True):
        from collections import Counter

        manager = self.model_runner.cudagraph_manager
        if enabled:
            if hasattr(self, "_v152_original_replay"):
                raise RuntimeError("Graph trace already enabled")
            self._v152_graph_counts = Counter()
            self._v152_dispatch_counts = Counter()
            self._v152_original_replay = manager.run_fullgraph
            self._v152_original_dispatch = manager.dispatch

            def recorded(desc):
                self._v152_graph_counts[str(desc)] += 1
                return self._v152_original_replay(desc)

            manager.run_fullgraph = recorded

            def dispatch(num_reqs, num_tokens, uniform_token_count):
                desc = self._v152_original_dispatch(
                    num_reqs, num_tokens, uniform_token_count
                )
                key = (
                    str((num_reqs, num_tokens, uniform_token_count))
                    + " -> "
                    + str(desc)
                )
                self._v152_dispatch_counts[key] += 1
                return desc

            manager.dispatch = dispatch
            return {"rank": self.rank, "enabled": True}
        manager.run_fullgraph = self._v152_original_replay
        manager.dispatch = self._v152_original_dispatch
        del self._v152_original_replay
        del self._v152_original_dispatch
        return {
            "rank": self.rank,
            "counts": dict(self._v152_graph_counts),
            "dispatch_counts": dict(self._v152_dispatch_counts),
        }

    def v152_start_profile(self):
        import torch

        self._v152_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        )
        self._v152_profiler.start()
        return {"rank": self.rank, "active": True}

    def v152_stop_profile(self, label="decode"):
        from pathlib import Path

        self._v152_profiler.stop()
        destination = Path("/profiles") / f"{label}-rank{self.rank}.json"
        self._v152_profiler.export_chrome_trace(str(destination))
        del self._v152_profiler
        return {"rank": self.rank, "trace": str(destination)}

    def v152_runtime_configuration(self):
        import os

        cache = self.vllm_config.cache_config
        sampler = getattr(self.model_runner, "rejection_sampler", None)
        graphs = getattr(sampler, "_sm70_dflash2_rejection_graphs", {})
        return {
            "rank": self.rank,
            "cache": {
                key: getattr(cache, key)
                for key in (
                    "partial_prefix_cache",
                    "prefix_cache_retention_interval",
                    "block_size",
                    "mamba_block_size",
                    "cache_dtype",
                )
            },
            "max_num_seqs": self.vllm_config.scheduler_config.max_num_seqs,
            "async_scheduling": self.vllm_config.scheduler_config.async_scheduling,
            "target_capture_sizes": self.model_runner.cudagraph_manager._capture_sizes,
            "rejection_graphs": len(graphs),
            "nccl_environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith("NCCL_")
            },
        }
