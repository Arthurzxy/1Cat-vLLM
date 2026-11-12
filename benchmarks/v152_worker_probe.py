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

    def _v152_request_snapshot(self, request_id):
        import hashlib

        import torch

        runner = self.model_runner
        states = runner.req_states
        slot = states.req_id_to_index.get(request_id)
        if slot is None:
            matches = [
                key for key in states.req_id_to_index if key.startswith(request_id)
            ]
            if len(matches) == 1:
                request_id = matches[0]
                slot = states.req_id_to_index[request_id]
        if slot is None:
            return None
        groups = runner.kv_cache_config.kv_cache_groups
        group_id = next(
            index
            for index, group in enumerate(groups)
            if type(group.kv_cache_spec).__name__ == "MambaSpec"
        )
        block_count = int(runner.block_tables.num_blocks.np[group_id, slot])
        blocks = (
            runner.block_tables.block_tables[group_id]
            .gpu[slot, :block_count]
            .cpu()
            .tolist()
        )
        live_blocks = [block for block in blocks if block != 0]
        result = {
            "request_id": request_id,
            "slot": slot,
            "prompt_tokens": int(states.prompt_len.np[slot]),
            "computed_tokens_cpu": int(states.num_computed_tokens_np[slot]),
            "computed_tokens_gpu": int(states.num_computed_tokens.gpu[slot].cpu()),
            "total_tokens_gpu": int(states.total_len.gpu[slot].cpu()),
            "last_sampled": states.last_sampled_tokens[slot].cpu().tolist(),
            "draft_tokens": states.draft_tokens[slot].cpu().tolist(),
            "mamba_blocks": blocks,
        }
        if live_blocks:
            layer_name = groups[group_id].layer_names[0]
            layer = self.vllm_config.compilation_config.static_forward_context[
                layer_name
            ]
            cache = layer.kv_cache
            model_state = runner.model_state
            assert model_state._align_mode, "State probe requires the active align path"
            state_col = int(model_state._mamba_state_idx_gpu[slot].cpu())
            accepted = int(model_state.num_accepted_tokens_gpu[slot].cpu())
            token_bias = max(accepted - 1, 0)
            ssm_col = (
                state_col
                if model_state._spec_replay is not None
                else state_col + token_bias
            )
            assert 0 <= state_col < len(blocks) and 0 <= ssm_col < len(blocks)
            physical = {"conv": blocks[state_col], "ssm": blocks[ssm_col]}
            tensors = {}
            for name, tensor in zip(("conv", "ssm"), cache):
                value = tensor[physical[name]].detach().contiguous().cpu()
                if name == "conv":
                    channels = layer.conv1d.weight.shape[0]
                    history = layer.conv1d.weight.shape[-1] - 1
                    time_axis = 0 if value.shape[-1] == channels else 1
                    assert value.shape[1 - time_axis] == channels
                    value = value.narrow(time_axis, token_bias, history).contiguous()
                tensors[name] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    "nonzero": int(torch.count_nonzero(value)),
                    "finite": bool(torch.isfinite(value).all()),
                    "sum": float(value.double().sum()),
                    "max_abs": float(value.abs().max()),
                }
            result["gdn_state"] = {
                "layer": layer_name,
                "physical_blocks": physical,
                "running_state_column": state_col,
                "accepted_tokens": accepted,
                "temporal_column": ssm_col,
                "conv_token_bias": token_bias,
                "scope": (
                    "Align-mode next-input state selected by the production "
                    "accepted-slot contract. GPU acceptance can extend past a "
                    "client max-token truncation."
                ),
                "tensors": tensors,
            }
        return result

    def v152_begin_request_state_trace(self):
        import time

        if getattr(self, "_v152_request_trace_installed", False):
            self._v152_request_events.clear()
            return {"rank": self.rank, "enabled": True}
        self._v152_request_trace_installed = True
        self._v152_request_events = []
        states = self.model_runner.req_states
        model_state = self.model_runner.model_state
        original_add = states.add_request
        original_remove = states.remove_request
        original_model_add = model_state.add_request

        def append(event):
            event["sequence"] = len(self._v152_request_events)
            event["time"] = time.monotonic()
            self._v152_request_events.append(event)

        def add(req_id, *args, **kwargs):
            original_add(req_id, *args, **kwargs)
            slot = states.req_id_to_index[req_id]
            append(
                {
                    "event": "add",
                    "request_id": req_id,
                    "slot": slot,
                    "initial_computed_tokens": int(states.num_computed_tokens_np[slot]),
                    "prompt_tokens": int(states.prompt_len.np[slot]),
                    "draft_zero": bool((states.draft_tokens[slot] == 0).all().cpu()),
                }
            )

        def remove(req_id):
            snapshot = self._v152_request_snapshot(req_id)
            slot = original_remove(req_id)
            if slot is not None:
                append(
                    {
                        "event": "remove",
                        "request_id": req_id,
                        "slot": slot,
                        "snapshot": snapshot,
                        "free_indices": list(states.free_indices),
                    }
                )
            return slot

        def model_add(req_index, new_req_data):
            original_model_add(req_index, new_req_data)
            append(
                {
                    "event": "model_add",
                    "slot": req_index,
                    "request_id": new_req_data.req_id,
                    "accepted_neutral": int(
                        model_state.num_accepted_tokens_gpu[req_index].cpu()
                    ),
                    "align_state_index": int(
                        model_state._mamba_state_idx_gpu[req_index].cpu()
                    )
                    if getattr(model_state, "_align_mode", False)
                    else None,
                }
            )

        states.add_request = add
        states.remove_request = remove
        model_state.add_request = model_add
        return {"rank": self.rank, "enabled": True}

    def v152_request_state_trace(self, request_id=None):
        states = self.model_runner.req_states
        return {
            "rank": self.rank,
            "requests": dict(states.req_id_to_index),
            "free_indices": list(states.free_indices),
            "events": list(getattr(self, "_v152_request_events", [])),
            "active_snapshot": self._v152_request_snapshot(request_id)
            if request_id is not None
            else None,
        }
