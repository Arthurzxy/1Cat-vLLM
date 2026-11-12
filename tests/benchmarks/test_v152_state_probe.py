# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-state observers preserve the wrapped keyword-call contracts."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def test_request_observer_preserves_production_keyword_names():
    root = Path(
        os.environ.get("V152_TEST_REPO_ROOT", Path(__file__).resolve().parents[2])
    )
    spec = importlib.util.spec_from_file_location(
        "v152_probe", root / "benchmarks/v152_worker_probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class States:
        def __init__(self):
            self.req_id_to_index = {}
            self.free_indices = list(range(8))
            self.num_computed_tokens_np = np.zeros(8, dtype=np.int32)
            self.prompt_len = SimpleNamespace(np=np.zeros(8, dtype=np.int32))
            self.draft_tokens = torch.full((8, 7), 91, dtype=torch.int64)

        def add_request(
            self, req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens
        ):
            index = self.free_indices.pop()
            self.req_id_to_index[req_id] = index
            self.num_computed_tokens_np[index] = num_computed_tokens
            self.prompt_len.np[index] = prompt_len
            self.draft_tokens[index].zero_()

        def remove_request(self, req_id):
            index = self.req_id_to_index.pop(req_id, None)
            if index is not None:
                self.free_indices.append(index)
            return index

    class ModelState:
        _align_mode = True

        def __init__(self):
            self.num_accepted_tokens_gpu = torch.zeros(8, dtype=torch.int32)
            self._mamba_state_idx_gpu = torch.zeros(8, dtype=torch.int32)

        def add_request(self, req_index, new_req_data):
            self.num_accepted_tokens_gpu[req_index] = 1
            self._mamba_state_idx_gpu[req_index] = -2

    states = States()
    model = ModelState()
    worker = SimpleNamespace(
        rank=0, model_runner=SimpleNamespace(req_states=states, model_state=model)
    )
    worker._v152_request_snapshot = lambda request_id: {"request_id": request_id}
    module.LongctxWorkerProbe.v152_begin_request_state_trace(worker)
    states.add_request(
        req_id="old",
        prompt_len=2,
        all_token_ids=[1, 2],
        num_computed_tokens=0,
        max_tokens=32,
    )
    model.add_request(req_index=7, new_req_data=SimpleNamespace(req_id="old"))
    assert states.remove_request(req_id="old") == 7
    states.add_request(
        req_id="new",
        prompt_len=2,
        all_token_ids=[3, 4],
        num_computed_tokens=0,
        max_tokens=32,
    )
    model.add_request(req_index=7, new_req_data=SimpleNamespace(req_id="new"))
    events = worker._v152_request_events
    assert [row["event"] for row in events] == [
        "add",
        "model_add",
        "remove",
        "add",
        "model_add",
    ]
    assert all(row["slot"] == 7 for row in events)
    assert events[0]["draft_zero"] and events[3]["draft_zero"]
    assert events[4]["request_id"] == "new"
    assert events[4]["accepted_neutral"] == 1
    assert events[4]["align_state_index"] == -2


def test_snapshot_selects_accepted_temporal_and_logical_conv_state():
    root = Path(
        os.environ.get("V152_TEST_REPO_ROOT", Path(__file__).resolve().parents[2])
    )
    spec = importlib.util.spec_from_file_location(
        "v152_probe", root / "benchmarks/v152_worker_probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spec_type = type("MambaSpec", (), {})
    groups = [SimpleNamespace(kv_cache_spec=spec_type(), layer_names=["layer"])]
    state = SimpleNamespace(
        req_id_to_index={"request-full-id": 7},
        prompt_len=SimpleNamespace(np=np.full(8, 2, dtype=np.int32)),
        num_computed_tokens_np=np.full(8, 9, dtype=np.int32),
        num_computed_tokens=SimpleNamespace(gpu=torch.full((8,), 10)),
        total_len=SimpleNamespace(gpu=torch.full((8,), 11)),
        last_sampled_tokens=torch.ones((8, 1), dtype=torch.int64),
        draft_tokens=torch.zeros((8, 7), dtype=torch.int64),
    )
    conv = torch.stack(
        [torch.full((10, 2), float(index), dtype=torch.float16) for index in range(9)]
    )
    ssm = torch.stack(
        [torch.full((2, 2), float(index), dtype=torch.float32) for index in range(9)]
    )
    layer = SimpleNamespace(
        kv_cache=(conv, ssm), conv1d=SimpleNamespace(weight=torch.empty((2, 1, 4)))
    )
    table = torch.arange(1, 9, dtype=torch.int32).expand(8, -1)
    runner = SimpleNamespace(
        req_states=state,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(np=np.full((1, 8), 8)),
            block_tables=[SimpleNamespace(gpu=table)],
        ),
        model_state=SimpleNamespace(
            _align_mode=True,
            _mamba_state_idx_gpu=torch.zeros(8, dtype=torch.int32),
            num_accepted_tokens_gpu=torch.full((8,), 3),
            _spec_replay=None,
        ),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={"layer": layer})
        ),
    )
    result = module.LongctxWorkerProbe._v152_request_snapshot(worker, "request")
    gdn = result["gdn_state"]
    assert gdn["physical_blocks"] == {"conv": 1, "ssm": 3}
    assert gdn["conv_token_bias"] == 2 and gdn["temporal_column"] == 2
    assert gdn["tensors"]["conv"]["shape"] == [3, 2]
    assert gdn["tensors"]["conv"]["sum"] == 6
    assert gdn["tensors"]["ssm"]["sum"] == 12
