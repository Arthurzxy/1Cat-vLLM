# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from next_qualification_worker import QualificationWorker


class EfficiencyWorker(QualificationWorker):
    def v152_efficiency_configuration(self):
        import time

        scheduler = self.vllm_config.scheduler_config
        cache = self.model_runner.kv_cache_config
        return {
            "rank": self.rank,
            "clock": {"epoch_s": time.time(), "monotonic_s": time.monotonic()},
            "scheduler": {
                name: getattr(scheduler, name)
                for name in (
                    "async_scheduling",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "long_prefill_token_threshold",
                    "mixed_prefill_step_latency_ms",
                    "mixed_prefill_min_tokens",
                    "enable_chunked_prefill",
                )
            },
            "cache": self.v152_runtime_configuration()["cache"],
            "geometry": {
                "logical_blocks": cache.num_blocks,
                "device_state_blocks": cache.device_state_blocks,
                "device_state_group_ids": list(cache.device_state_group_ids),
                "groups": [
                    {
                        "type": type(group.kv_cache_spec).__name__,
                        "block_size": group.kv_cache_spec.block_size,
                        "page_size_bytes": group.kv_cache_spec.page_size_bytes,
                        "layers": len(group.layer_names),
                    }
                    for group in cache.kv_cache_groups
                ],
            },
        }
