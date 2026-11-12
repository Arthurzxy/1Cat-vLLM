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
            self._v152_aligned_hit_tokens = result[1]
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


def install_final_lookup():
    directory = os.environ.get("V152_CACHE_TRACE")
    if not directory:
        return
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original = KVCacheManager.get_computed_blocks

    @functools.wraps(original)
    def lookup(self, request):
        result = original(self, request)
        entry = request.partial_prefix_entry
        cache = self.partial_cache
        record = {
            "stage": "manager_final",
            "time": time.time(),
            "pid": os.getpid(),
            "request_id": request.request_id,
            "block_aligned_hit_tokens": getattr(
                self.coordinator, "_v152_aligned_hit_tokens", None
            ),
            "final_hit_tokens": result[1],
            "shared_prefix_boundary": request.shared_prefix_boundary,
            "partial_block_size": getattr(self, "partial_block_size", None),
            "hash_block_size": getattr(self, "hash_block_size", None),
            "partial_entry_length": entry.length if entry else None,
            "partial_copy_last": entry.copy_last if entry else None,
            "partial_ready_step": entry.ready_step if entry else None,
            "partial_entry_block_ids": [
                [block.block_id for block in group] for group in entry.blocks
            ]
            if entry
            else None,
            "partial_cache_entries": len(cache.entries) if cache else 0,
            "partial_cache_owned_blocks": cache.num_blocks if cache else 0,
            "partial_cache_block_limit": cache.max_blocks if cache else 0,
        }
        with (Path(directory) / f"lookups-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return result

    KVCacheManager.get_computed_blocks = lookup


install_final_lookup()


def install_pool_observation():
    directory = os.environ.get("V152_CACHE_TRACE")
    if not directory:
        return
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    def snapshot(self, stage, request_id, detailed=False, allocated=None):
        pool = self.block_pool
        queue = pool.free_block_queue
        if hasattr(queue, "state_blocks"):
            state_end = queue.state_blocks
            partitions = {
                "state": {
                    "capacity_blocks": state_end - 1,
                    "free_or_evictable_blocks": queue.state.num_free_blocks,
                },
                "history": {
                    "capacity_blocks": pool.num_gpu_blocks - state_end,
                    "free_or_evictable_blocks": queue.history.num_free_blocks,
                },
            }
            if detailed:
                partitions["state"]["cached_physical_blocks"] = sum(
                    block.block_hash is not None for block in pool.blocks[1:state_end]
                )
                partitions["history"]["cached_physical_blocks"] = sum(
                    block.block_hash is not None for block in pool.blocks[state_end:]
                )
        else:
            partitions = {
                "shared": {
                    "capacity_blocks": pool.num_gpu_blocks - 1,
                    "free_or_evictable_blocks": pool.get_num_free_blocks(),
                }
            }
            if detailed:
                partitions["shared"]["cached_physical_blocks"] = sum(
                    block.block_hash is not None for block in pool.blocks[1:]
                )
        for partition in partitions.values():
            partition["pinned_blocks"] = (
                partition["capacity_blocks"] - partition["free_or_evictable_blocks"]
            )
        groups = []
        for index, manager in enumerate(self.coordinator.single_type_managers):
            blocks = [
                block
                for sequence in manager.req_to_blocks.values()
                for block in sequence
                if not block.is_null
            ]
            groups.append(
                {
                    "group": index,
                    "manager": type(manager).__name__,
                    "active_requests": len(manager.req_to_blocks),
                    "active_block_references": len(blocks),
                    "active_unique_physical_blocks": len(
                        {block.block_id for block in blocks}
                    ),
                    "free_or_evictable_in_group_partition": pool.get_num_free_blocks(
                        index
                    ),
                }
            )
        partial = self.partial_cache
        record = {
            "stage": stage,
            "time": time.time(),
            "pid": os.getpid(),
            "request_id": request_id,
            "allocation_succeeded": allocated,
            "partitions": partitions,
            "groups": groups,
            "cached_hash_keys": len(pool.cached_block_hash_to_block),
            "partial_entries": len(partial.entries) if partial else 0,
            "partial_owned_blocks": partial.num_blocks if partial else 0,
            "partial_block_limit": partial.max_blocks if partial else 0,
            "partial_entry_lengths": sorted(
                entry.length for entry in partial.entries.values()
            )
            if partial
            else [],
            "count_definition": (
                "Free includes evictable cached blocks. Cached physical counts "
                "overlap free/pinned status. Group active references may share "
                "physical IDs."
            ),
        }
        with (Path(directory) / f"pools-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    original_lookup = KVCacheManager.get_computed_blocks
    original_allocate = KVCacheManager.allocate_slots
    original_free = KVCacheManager.free

    @functools.wraps(original_lookup)
    def lookup(self, request):
        result = original_lookup(self, request)
        snapshot(self, "lookup", request.request_id, detailed=True)
        return result

    @functools.wraps(original_allocate)
    def allocate(self, request, *args, **kwargs):
        result = original_allocate(self, request, *args, **kwargs)
        snapshot(self, "allocate", request.request_id, allocated=result is not None)
        return result

    @functools.wraps(original_free)
    def free(self, request):
        result = original_free(self, request)
        snapshot(self, "free", request.request_id, detailed=True)
        return result

    KVCacheManager.get_computed_blocks = lookup
    KVCacheManager.allocate_slots = allocate
    KVCacheManager.free = free


install_pool_observation()
