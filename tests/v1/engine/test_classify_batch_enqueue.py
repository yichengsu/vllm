# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.engine.protocol import PoolingGroupItem
from vllm.pooling_params import PoolingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import RequestStatus


def make_core():
    core = object.__new__(EngineCoreProc)
    core.shutdown_state = EngineShutdownState.RUNNING
    core.scheduler = create_scheduler(max_num_seqs=5)
    core._send_error_outputs_to_client = Mock()
    return core


def test_batch_visibility_and_split():
    core = make_core()
    requests = create_requests(num_requests=12)
    core._handle_client_request(
        EngineCoreRequestType.ADD_GROUP, [(r, 0) for r in requests]
    )
    assert list(core.scheduler.waiting) == requests
    for start in (0, 5, 10):
        step = core.scheduler.schedule()
        assert list(step.num_scheduled_tokens) == [
            r.request_id for r in requests[start : start + 5]
        ]
        core.scheduler.finish_requests(
            list(step.num_scheduled_tokens), RequestStatus.FINISHED_STOPPED
        )


@pytest.mark.parametrize("collision", [False, True])
def test_batch_rollback_preserves_other_requests(collision):
    core = make_core()
    batch = create_requests(num_requests=3)
    prior = create_requests(
        num_requests=1, req_ids=[batch[1].request_id if collision else "prior"]
    )[0]
    core.scheduler.add_request(prior)

    def add(req, wave):
        core.scheduler.add_request(req)
        if req is batch[1]:
            raise ValueError("injected failure after insertion")

    core.add_request = add
    core._handle_client_request(
        EngineCoreRequestType.ADD_GROUP, [(r, 0) for r in batch]
    )
    assert list(core.scheduler.waiting) == [prior]
    assert core.scheduler.requests == {prior.request_id: prior}
    core._send_error_outputs_to_client.assert_called_once_with(
        [r.request_id for r in batch], batch[0].client_index
    )


@pytest.mark.asyncio
async def test_disconnect_during_submission_is_worker_scoped(monkeypatch):
    # Even identical client-supplied IDs and random bits must be safe across
    # API workers: the ABORT wire message identifies requests only by ID.
    monkeypatch.setattr("vllm.v1.engine.async_llm.random_uuid", lambda: "a" * 32)
    params = PoolingParams(task="classify")
    items = [
        PoolingGroupItem(
            f"shared-{i}", {"type": "token", "prompt_token_ids": [1]}, params
        )
        for i in range(2)
    ]
    pending = []
    for client_index in (0, 1):
        engine = Mock()
        engine.errored = False
        engine.model_config.is_encoder_decoder = False
        engine.engine_core.client_index = client_index
        engine.get_supported_tasks = AsyncMock(return_value=("classify",))
        engine.input_processor.assign_request_id = InputProcessor.assign_request_id
        engine.input_processor.process_inputs.side_effect = lambda rid, *a, **kw: (
            SimpleNamespace(
                request_id=rid,
                external_req_id=None,
                prompt_token_ids=[1],
                params=params,
            )
        )
        states: dict[str, SimpleNamespace] = {}
        engine.output_processor.request_states = states
        engine.output_processor.has_request.side_effect = states.__contains__
        engine.output_processor.add_request.side_effect = (
            lambda req, text, parent, idx, q, states=states: states.update(
                {req.request_id: SimpleNamespace(queue=q)}
            )
        )
        sending = asyncio.Event()

        async def send(requests, sending=sending):
            sending.set()
            await asyncio.Future()

        engine.engine_core.add_requests_async = send
        engine.abort = AsyncMock()
        task = asyncio.create_task(
            anext(AsyncLLM.encode_group(engine, items, "shared"))
        )
        await asyncio.wait_for(sending.wait(), 5)
        pending.append((engine, task, set(states)))

    (_, first_task, first_ids), (second_engine, second_task, second_ids) = pending
    try:
        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task
        second_engine.abort.assert_awaited_once()
        cancelled_ids = set(second_engine.abort.await_args.args[0])
        assert cancelled_ids == second_ids
        assert first_ids.isdisjoint(cancelled_ids)
    finally:
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)
