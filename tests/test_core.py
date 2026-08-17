"""Harness behaviour tests — no network. A scripted fake LLM drives the loop, which
is exactly what the `llm` injection point on Agent is for.

These cover the failure modes the framework promises to handle:
  - normal run: tool called, result observed, final answer returned
  - unknown tool name -> error observation fed back, run survives
  - tool that raises -> error observation fed back, run survives
  - malformed JSON args -> tolerated as {}
  - max_steps hit -> forced tool-free final answer, stopped_reason="max_steps"
  - shared state visible to tools and caller
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from agentloop import NO_ARGS, Agent, Tool


# ── a scripted fake of the OpenAI assistant message ─────────────────────────────
@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMsg:
    content: str = ""
    tool_calls: list = field(default_factory=list)


def scripted_llm(script: list[FakeMsg]):
    """Returns an llm callable that pops one scripted reply per call."""
    replies = list(script)

    async def llm(messages, tools):
        if not replies:  # out of script -> behave like a model giving a final answer
            return FakeMsg(content="(scripted fallback final)")
        return replies.pop(0)

    return llm


def tc(name: str, arguments: str = "{}", id_: str = "tc1") -> FakeToolCall:
    return FakeToolCall(id=id_, function=FakeFunction(name=name, arguments=arguments))


# ── tools used by the tests ─────────────────────────────────────────────────────
async def add_tool(state: dict, a: int = 0, b: int = 0, **_):
    state["sum"] = a + b
    return {"sum": a + b}


async def boom_tool(state: dict, **_):
    raise RuntimeError("kaboom")


ADD = Tool("add", "Add two numbers.", {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}, add_tool)
BOOM = Tool("boom", "Always fails.", NO_ARGS, boom_tool)


# ── tests ───────────────────────────────────────────────────────────────────────
async def test_normal_run_calls_tool_then_finishes():
    llm = scripted_llm([
        FakeMsg(tool_calls=[tc("add", '{"a": 2, "b": 3}')]),
        FakeMsg(content="The sum is 5."),
    ])
    agent = Agent("t", "sys", [ADD], llm=llm)
    res = await agent.run("add 2 and 3")
    assert res.stopped_reason == "final"
    assert res.output == "The sum is 5."
    assert res.tool_calls == 1
    assert res.state["sum"] == 5
    kinds = [s.kind for s in res.steps]
    assert kinds == ["tool_call", "final"]


async def test_unknown_tool_is_an_observation_not_a_crash():
    llm = scripted_llm([
        FakeMsg(tool_calls=[tc("does_not_exist")]),
        FakeMsg(content="ok, recovered"),
    ])
    agent = Agent("t", "sys", [ADD], llm=llm)
    res = await agent.run("go")
    assert res.stopped_reason == "final"
    tool_steps = [s for s in res.steps if s.kind == "tool_call"]
    assert "unknown tool" in tool_steps[0].content


async def test_raising_tool_is_fed_back_as_error():
    llm = scripted_llm([
        FakeMsg(tool_calls=[tc("boom")]),
        FakeMsg(content="saw the error, giving up gracefully"),
    ])
    agent = Agent("t", "sys", [BOOM], llm=llm)
    res = await agent.run("go")
    assert res.stopped_reason == "final"
    tool_steps = [s for s in res.steps if s.kind == "tool_call"]
    assert "ERROR running boom" in tool_steps[0].content
    assert "kaboom" in tool_steps[0].content


async def test_malformed_json_args_tolerated():
    llm = scripted_llm([
        FakeMsg(tool_calls=[tc("add", "{not json at all")]),
        FakeMsg(content="done"),
    ])
    agent = Agent("t", "sys", [ADD], llm=llm)
    res = await agent.run("go")
    # add ran with defaults (a=0, b=0) instead of crashing the run
    assert res.state["sum"] == 0
    assert res.stopped_reason == "final"


async def test_max_steps_forces_final_answer():
    # The model calls a tool on every turn and never answers -> harness must cut it
    # off at max_steps and force a tool-free final. Script: exactly max_steps tool
    # turns, then the reply to the forced "answer now" prompt.
    endless = [FakeMsg(tool_calls=[tc("add", '{"a": 1, "b": 1}')]) for _ in range(3)]
    endless.append(FakeMsg(content="forced final"))
    llm = scripted_llm(endless)
    agent = Agent("t", "sys", [ADD], max_steps=3, llm=llm)
    res = await agent.run("go")
    assert res.stopped_reason == "max_steps"
    assert res.output  # never empty
    assert res.tool_calls == 3


async def test_explicit_model_is_sent_to_the_provider():
    """Agent(model=...) is an instruction. Before this was fixed the provider chain
    overwrote it with the provider's own model and the caller was never told."""
    import agentloop.core as core

    os.environ["AGENTLOOP_BASE_URL"] = "http://127.0.0.1:9/v1"
    os.environ["AGENTLOOP_API_KEY"] = "x"
    os.environ["AGENTLOOP_MODEL"] = "provider-model"
    core.reset_providers()
    try:
        sent: list[str] = []

        class FakeCompletions:
            async def create(self, **kwargs):
                sent.append(kwargs["model"])
                msg = SimpleNamespace(content="ok", tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        for prov in core._get_providers():
            prov.client.chat.completions = FakeCompletions()

        await core.Agent("t", "sys", [], model="asked-for-this-one").run("go")
        assert sent == ["asked-for-this-one"]

        sent.clear()
        await core.Agent("t", "sys", []).run("go")
        assert sent == ["provider-model"]  # no override -> provider decides
    finally:
        for key in ("AGENTLOOP_BASE_URL", "AGENTLOOP_API_KEY", "AGENTLOOP_MODEL"):
            os.environ.pop(key, None)
        core.reset_providers()


def test_salvage_json_survives_prose_and_takes_the_answer():
    """A greedy {.*} span returned {} for anything with two objects in it."""
    from agentloop.core import _salvage_json

    assert _salvage_json('Here is the result: {"score": 85}') == {"score": 85}
    assert _salvage_json('{"score": 85}\nLet me know.') == {"score": 85}
    # example first, answer second: the answer must win
    assert _salvage_json('Example: {"score": 0}, my answer: {"score": 85}') == {"score": 85}
    assert _salvage_json('{"outer": {"inner": 1}}') == {"outer": {"inner": 1}}
    assert _salvage_json("no json here at all") is None


def test_tool_rejects_reserved_state_argument():
    """Tools are called as fn(state=..., **model_args), so a schema property named
    'state' collides and the tool can never run. Fail at construction, not at call."""
    with pytest.raises(ValueError, match="reserved"):
        Tool("bad", "clashes", {
            "type": "object",
            "properties": {"state": {"type": "string"}},
        }, add_tool)


async def test_state_is_shared_between_tools_and_caller():
    async def write_tool(state: dict, **_):
        state["note"] = "left by tool"
        return "ok"

    llm = scripted_llm([
        FakeMsg(tool_calls=[tc("write")]),
        FakeMsg(content="done"),
    ])
    agent = Agent("t", "sys", [Tool("write", "writes state", NO_ARGS, write_tool)], llm=llm)
    caller_state: dict = {"preexisting": True}
    res = await agent.run("go", state=caller_state)
    assert caller_state["note"] == "left by tool"
    assert res.state is caller_state