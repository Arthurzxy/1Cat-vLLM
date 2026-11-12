# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU-only scheduler observation. Never used for formal latency."""

import json
import os
import time
from pathlib import Path

if os.environ.get("V152_SCHEDULER_TRACE"):
    from vllm.v1.core.sched.scheduler import Scheduler

    original_schedule = Scheduler.schedule
    original_update = Scheduler.update_from_output
    logs = {}

    def emit(value):
        pid = os.getpid()
        if pid not in logs:
            folder = Path(os.environ["V152_SCHEDULER_TRACE"])
            folder.mkdir(exist_ok=True)
            logs[pid] = (folder / f"scheduler-{pid}.jsonl").open("a", buffering=1)
        logs[pid].write(json.dumps(value) + "\n")

    def schedule(self):
        now = time.monotonic()
        last = getattr(self, "_next_observed_outputs", {})
        running = [
            {
                "request_id": request.request_id,
                "prompt_tokens": request.num_prompt_tokens,
                "computed_tokens": request.num_computed_tokens,
                "output_tokens": request.num_output_tokens,
                "output_placeholders": request.num_output_placeholders,
                "last_output_age_s": now - last[request.request_id]
                if request.request_id in last
                else None,
            }
            for request in self.running
        ]
        result = original_schedule(self)
        emit(
            {
                "event": "schedule",
                "at": now,
                "class": type(self).__name__,
                "step": self.current_step,
                "running_before": running,
                "scheduled_tokens": result.num_scheduled_tokens,
                "mixed_prefill_tokens": result.mixed_prefill_tokens,
                "mixed_decode_tokens": result.mixed_decode_tokens,
                "mixed_budget": result.mixed_prefill_budget,
                "controller_tokens": self.mixed_prefill_budget.tokens,
            }
        )
        return result

    def update(self, scheduled, output):
        result = original_update(self, scheduled, output)
        now = time.monotonic()
        if not hasattr(self, "_next_observed_outputs"):
            self._next_observed_outputs = {}
        emitted = []
        for client in result.values():
            for request in client.outputs:
                if request.new_token_ids:
                    self._next_observed_outputs[request.request_id] = now
                    emitted.append(
                        {
                            "request_id": request.request_id,
                            "committed_tokens": len(request.new_token_ids),
                        }
                    )
        sample = output.mixed_prefill_timing
        emit(
            {
                "event": "update",
                "at": now,
                "step": self.current_step,
                "emitted": emitted,
                "controller_tokens": self.mixed_prefill_budget.tokens,
                "feedback": {
                    name: getattr(sample, name)
                    for name in (
                        "prefill_tokens",
                        "decode_tokens",
                        "elapsed_ms",
                        "budget_tokens",
                        "cuda_graph",
                    )
                }
                if sample is not None
                else None,
            }
        )
        return result

    Scheduler.schedule = schedule
    Scheduler.update_from_output = update
