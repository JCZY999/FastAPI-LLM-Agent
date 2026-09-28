import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent import AgentFailure, execute_tool, run_agent


def response(*, calls=None, text="", status="completed"):
    return SimpleNamespace(status=status, output=calls or [], output_text=text)


def call(name="lookup_lesson", arguments='{"topic":"pydantic"}'):
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id="call_1")


def test_tool_cycle_preserves_call_id_and_returns_answer():
    create = AsyncMock(side_effect=[response(calls=[call()]), response(text="Pydantic validates data.")])
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    answer, used = asyncio.run(run_agent(client, "test-model", "Explain Pydantic"))
    assert answer == "Pydantic validates data."
    assert used == ["lookup_lesson"]
    sent = create.call_args.kwargs["input"][-1]
    assert sent["call_id"] == "call_1"
    assert json.loads(sent["output"])["topic"] == "pydantic"
    assert create.call_args.kwargs["store"] is False


@pytest.mark.parametrize("name,args", [
    ("run_shell", '{}'), ("lookup_lesson", 'not-json'),
    ("lookup_lesson", '{"topic":"secrets"}'),
    ("lookup_lesson", '{"topic":"fastapi","extra":true}'),
])
def test_invalid_tools_are_rejected(name, args):
    assert "error" in json.loads(execute_tool(name, args))


def test_step_limit():
    create = AsyncMock(return_value=response(calls=[call()]))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(AgentFailure, match="step limit"):
        asyncio.run(run_agent(client, "test-model", "loop"))
    assert create.call_count == 4


@pytest.mark.parametrize("result", [response(), response(status="incomplete"), response(calls=[call()] * 5)])
def test_invalid_provider_responses(result):
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=result)))
    with pytest.raises(AgentFailure):
        asyncio.run(run_agent(client, "test-model", "hello"))
