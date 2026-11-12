# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from next_qualification_worker import QualificationWorker


class EfficiencyWorker(QualificationWorker):
    def v152_efficiency_configuration(self):
        scheduler = self.vllm_config.scheduler_config
        return {
            "rank": self.rank,
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
        }
