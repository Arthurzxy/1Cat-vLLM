# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent insertion oracle for rejection resampling and changed replay."""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as sampling
from vllm.v1.worker.gpu.spec_decode.dflash2 import sampler_graph


def production_insert_warps():
    tree = ast.parse(inspect.getsource(sampling.rejection_sample))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_insert_resampled_kernel"
    ]
    assert len(calls) == 1
    values = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    return ast.literal_eval(values["num_warps"]) if "num_warps" in values else 4


@pytest.mark.parametrize("max_width", range(1, 9))
@pytest.mark.parametrize("blocks", [1, 7, 243])
@pytest.mark.parametrize("execution", ["eager", "graph"])
def test_resample_insert_matches_independent_oracle(max_width, blocks, execution):
    requests = 17
    sentinel = -91234567
    widths = [1 + index % max_width for index in range(requests)]
    cumulative = [0]
    for width in widths:
        cumulative.append(cumulative[-1] + width)
    generator = torch.Generator().manual_seed(147)
    state_indices = torch.randperm(requests * 3, generator=generator)[:requests]
    temperatures = torch.tensor(
        [0.0 if index % 2 == 0 else 0.7 for index in range(requests * 3)]
    )
    expanded = [
        int(state_indices[row])
        for row, width in enumerate(widths)
        for _ in range(width)
    ]
    cu = torch.tensor(cumulative, dtype=torch.int32, device="cuda")
    mapping = torch.tensor(expanded, dtype=torch.int32, device="cuda")
    temperature = temperatures.to("cuda")
    initial = torch.full((requests, max_width + 4), sentinel, dtype=torch.int64)
    for row, width in enumerate(widths):
        initial[row, 1 : 1 + width] = torch.arange(width) + 400 + row * 16
    initial_device = initial.to("cuda")
    storage = torch.empty_like(initial_device)
    sampled = storage[:, 1 : 1 + max_width]
    count_storage = torch.full(
        (requests + 4,), sentinel, dtype=torch.int32, device="cuda"
    )
    counts = count_storage[1 : 1 + requests]
    input_counts = torch.empty_like(counts)
    candidates_cpu = (
        torch.arange(blocks, dtype=torch.int64)[None, :] * 1024
        + torch.arange(requests, dtype=torch.int64)[:, None]
    )
    candidates = candidates_cpu.to("cuda")
    maxima = torch.empty((requests, blocks), dtype=torch.float32, device="cuda")
    warps = production_insert_warps()

    def run():
        storage.copy_(initial_device)
        counts.copy_(input_counts)
        sampling._insert_resampled_kernel[(requests,)](
            sampled,
            sampled.stride(0),
            counts,
            candidates,
            candidates.stride(0),
            maxima,
            maxima.stride(0),
            blocks,
            cu,
            mapping,
            temperature,
            PADDED_RESAMPLE_NUM_BLOCKS=1 << (blocks - 1).bit_length(),
            num_warps=warps,
        )

    input_counts.zero_()
    maxima.zero_()
    run()
    graph = None
    if execution == "graph":
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

    # The host oracle only edits the original rejected/bonus position. Every
    # accepted count, ragged boundary, request state and winning block changes.
    for iteration in range(max_width * 4):
        accepted = [(iteration + row) % width for row, width in enumerate(widths)]
        winners = [(iteration * 13 + row * 19) % blocks for row in range(requests)]
        values = torch.full((requests, blocks), -1234.0)
        for row, winner in enumerate(winners):
            values[row, winner] = 10.0 + row
        input_counts.copy_(torch.tensor(accepted, dtype=torch.int32))
        maxima.copy_(values)
        expected = initial.clone()
        for row, count in enumerate(accepted):
            is_bonus = count == widths[row] - 1
            if temperatures[state_indices[row]] != 0 or is_bonus:
                expected[row, 1 + count] = candidates_cpu[row, winners[row]]
        if graph is None:
            run()
        else:
            graph.replay()
        torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            counts.cpu(),
            torch.tensor(accepted, dtype=torch.int32) + 1,
            rtol=0,
            atol=0,
        )
        expected_guards = torch.full((requests + 4,), sentinel, dtype=torch.int32)
        expected_guards[1 : 1 + requests] = torch.tensor(accepted) + 1
        torch.testing.assert_close(count_storage.cpu(), expected_guards, rtol=0, atol=0)


@pytest.mark.parametrize("route", ["ordinary", "dense", "dense-graph"])
@pytest.mark.parametrize("vocab", [31, 248320])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_actual_rejection_routes_with_independent_point_mass_oracle(
    route, vocab, use_fp64
):
    requests = 8 if route == "ordinary" else 3
    slots = requests + 4
    widths = list(range(1, 9)) if route == "ordinary" else [8] * requests
    cumulative = [0]
    for width in widths:
        cumulative.append(cumulative[-1] + width)
    rows = cumulative[-1]
    mapping_cpu = torch.arange(slots - 1, slots - 1 - requests, -1)
    mapping = mapping_cpu.to(device="cuda", dtype=torch.int32)
    expanded = torch.repeat_interleave(mapping, torch.tensor(widths, device="cuda"))
    local = torch.tensor(
        [position for width in widths for position in range(width)],
        dtype=torch.int32,
        device="cuda",
    )
    positions = local.long() + 8192
    cu = torch.tensor(cumulative, dtype=torch.int32, device="cuda")
    inputs = torch.full((rows,), 3, dtype=torch.int32, device="cuda")
    target = torch.full((rows, vocab), -float("inf"), device="cuda")
    draft = torch.full((slots, 7, vocab), -float("inf"), device="cuda")
    draft[:, :, 3] = 0.0
    temperature = torch.full((slots,), 0.7, device="cuda")
    temperature[::2] = 0.0
    seeds = torch.arange(slots, dtype=torch.int64, device="cuda") + 127

    def ordinary():
        return sampling.rejection_sample(
            target,
            draft,
            inputs,
            cu,
            positions,
            mapping,
            expanded,
            local,
            temperature,
            seeds,
            7,
            use_fp64=use_fp64,
        )

    dense = object.__new__(sampler_graph._RejectionGraph)
    dense.requests = requests
    dense.draft_logits = torch.empty_like(draft)
    dense.dense_pointer = torch.tensor(
        [draft.data_ptr()], dtype=torch.int64, device="cuda"
    )
    dense.mapping = mapping
    dense.expanded = expanded
    dense.request_indices = mapping_cpu.numpy()
    dense.positions = positions
    dense.inputs = inputs
    dense.local = local
    dense.cu = cu
    dense.fallback = lambda: target
    dense.rejection = SimpleNamespace(num_speculative_steps=7)
    # Inputs are already transformed point-mass target distributions, so the
    # sampler's parameter transform is the identity in this narrow fixture.
    dense.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperature),
            seeds=SimpleNamespace(gpu=seeds),
        ),
        use_fp64_gumbel=use_fp64,
        apply_sampling_params=lambda logits, *args: logits,
    )
    dense.output = torch.empty((requests, 8), dtype=torch.int64, device="cuda")
    dense.count = torch.empty(requests, dtype=torch.int32, device="cuda")

    def run():
        if route == "ordinary":
            return ordinary()
        dense._dense()
        return dense.output, dense.count

    target[:, 11] = 0.0
    run()
    graph = None
    if route == "dense-graph":
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, count = run()
    for iteration in range(16):
        target.fill_(-float("inf"))
        expected = []
        for row, width in enumerate(widths):
            rejected = (iteration + row) % width
            start = cumulative[row]
            for position in range(width):
                token = 11 if position == width - 1 else 3
                target[start + position, token] = 0.0
            if rejected < width - 1:
                target[start + rejected, 3] = -float("inf")
                target[start + rejected, 7] = 0.0
                expected.append([3] * rejected + [7])
            else:
                expected.append([3] * (width - 1) + [11])
        seeds.add_(17)
        if graph is None:
            output, count = run()
        else:
            graph.replay()
        actual_count = count.cpu().tolist()
        actual_output = output.cpu()
        assert actual_count == list(map(len, expected))
        for row, tokens in enumerate(expected):
            assert actual_output[row, : len(tokens)].tolist() == tokens
