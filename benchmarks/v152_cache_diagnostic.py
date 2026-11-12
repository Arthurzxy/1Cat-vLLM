# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-group prefix lookup audit for isolated cache experiments."""

import contextvars
import functools
import json
import os
import time
from pathlib import Path


def install():
    directory = os.environ.get("V152_CACHE_TRACE")
    if not directory:
        return
    from vllm.v1.core import kv_cache_coordinator as coordinator
    from vllm.v1.core import single_type_kv_cache_manager as managers

    active = contextvars.ContextVar("v152_cache_lookup", default=None)

    def manager_wrapper(method):
        @functools.wraps(method)
        def wrapped(cls, *args, **kwargs):
            result = method(cls, *args, **kwargs)
            rows = active.get()
            if rows is not None:
                spec = kwargs["kv_cache_spec"]
                rows.append(
                    {
                        "manager": cls.__name__,
                        "spec": type(spec).__name__,
                        "group_ids": kwargs["kv_cache_group_ids"],
                        "block_size": spec.block_size,
                        "alignment": kwargs["alignment_tokens"],
                        "max_length": kwargs["max_length"],
                        "drop_eagle": kwargs["drop_eagle_block"],
                        "returned_span_tokens": [
                            len(group) * spec.block_size for group in result
                        ],
                        "physical_blocks": [
                            sum(not block.is_null for block in group)
                            for group in result
                        ],
                    }
                )
            return result

        return classmethod(wrapped)

    for cls in vars(managers).values():
        if isinstance(cls, type) and issubclass(cls, managers.SingleTypeKVCacheManager):
            method = cls.__dict__.get("find_longest_cache_hit")
            if isinstance(method, classmethod) and not getattr(
                method.__func__, "__isabstractmethod__", False
            ):
                cls.find_longest_cache_hit = manager_wrapper(method.__func__)
    original = coordinator.HybridKVCacheCoordinator.find_longest_cache_hit

    @functools.wraps(original)
    def lookup(self, block_hashes, max_cache_hit_length):
        rows = []
        token = active.set(rows)
        try:
            result = original(self, block_hashes, max_cache_hit_length)
            record = {
                "time": time.time(),
                "pid": os.getpid(),
                "max_cache_hit_length": max_cache_hit_length,
                "intersection_tokens": result[1],
                "shared_prefix_boundary": self.shared_prefix_boundary,
                "groups": rows,
                "final_group_span_tokens": [
                    len(blocks) * group.kv_cache_spec.block_size
                    for blocks, group in zip(
                        result[0], self.kv_cache_config.kv_cache_groups
                    )
                ],
            }
            with (Path(directory) / f"lookups-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            return result
        finally:
            active.reset(token)

    coordinator.HybridKVCacheCoordinator.find_longest_cache_hit = lookup


install()
