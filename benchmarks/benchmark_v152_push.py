# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-rank fixed-input graph timings for TP4 push admission boundaries."""

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous, output):
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.config.collective import CollectiveNativeConfig
    from vllm.config.execution_policy import CommunicationPolicy
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=180),
    )
    policy = CommunicationPolicy(
        tp4_push=True,
        native=CollectiveNativeConfig(
            sm70_tp4_push_allreduce_concurrency=True,
            sm70_tp4_push_allreduce_small_messages=True,
        ),
    )
    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=4, communication=policy)
    )
    with set_current_vllm_config(config):
        communicator = CustomAllreduce(dist.group.WORLD, rank, max_size=2 * 1024 * 1024)
    assert not communicator.disabled and communicator.fully_connected
    results = []
    try:
        for rows in (8, 16, 32, 33, 40, 48, 56, 64, 65):
            torch.manual_seed(410 + rank)
            x = torch.randn((rows, 5120), device="cuda", dtype=torch.float16) * 0.1
            y = torch.empty_like(x)
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with communicator.capture(), torch.cuda.graph(graph):
                for _ in range(32):
                    communicator.all_reduce(x, out=y, registered=True)
            for _ in range(8):
                graph.replay()
            torch.accelerator.synchronize()
            samples = []
            for trial in range(5):
                dist.barrier()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(32):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / 1024)
            result = {
                "rows": rows,
                "samples_us": samples,
                "output_sha256": hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),
                "input_sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
            }
            results.append(result)
    finally:
        communicator.close()
        dist.destroy_process_group()
    Path(output + f"-rank{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "results": results,
                "scope": "fixed-input hot-L2 CUDA graph operator timings",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rendezvous = Path(args.output + "-rendezvous")
    if rendezvous.exists():
        raise FileExistsError(rendezvous)
    mp.spawn(worker, args=(rendezvous.as_uri(), args.output), nprocs=4, join=True)
