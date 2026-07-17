"""
agentloop core — a minimal, transparent agent harness.

WHAT MAKES THIS "AGENTIC" (vs. a normal LLM call or a fixed pipeline):
  - normal call:  input -> LLM -> output           (one shot)
  - pipeline:     YOU hardcode step order          (deterministic)
  - agent (here): you give the model a GOAL + TOOLS, and the *model* decides which
                  tool to call and when, in a loop, observing each result before
                  deciding the next move. ReAct: Reason -> Act -> Observe -> repeat.

The hard parts of agentic engineering all live here, on purpose (a framework would
hide them):
  - TOOL DESIGN ........ Tool dataclass: clear name/description/JSON-schema + helpful
                         errors fed back to the model instead of crashing the run.
  - HARNESS BEHAVIOUR .. the run() loop: call model -> run tools -> feed results back.
  - FAILURE MODES ...... max_steps (no infinite loops), per-tool try/except (a broken
                         tool becomes an observation, not a crash), bad-JSON args
                         tolerated, transient provider errors retried, forced final
                         answer when steps run out.
  - OBSERVABILITY ...... every model thought and tool call is recorded as a Step, so a
                         whole run can be inspected, replayed, or graded by evals.

No third-party agent library — this file *is* the framework.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

logger = logging.getLogger("agentloop")

# ── configuration ────────────────────────────────────────────────────────────────
# agentloop talks to any OpenAI-compatible endpoint. Configure with env vars:
#
#   AGENTLOOP_BASE_URL   e.g. https://api.groq.com/openai/v1  or  http://localhost:11434/v1
#   AGENTLOOP_API_KEY    key for that endpoint (local servers ignore it)
#   AGENTLOOP_MODEL      e.g. llama-3.3-70b-versatile, gpt-4o-mini, qwen2.5:7b-instruct
#
# Optional second provider (fallback chain, tried in order):
#
#   AGENTLOOP_BASE_URL_2 / AGENTLOOP_API_KEY_2 / AGENTLOOP_MODEL_2
#
# If nothing is set: OPENAI_API_KEY -> OpenAI gpt-4o-mini; otherwise a local
# Ollama server at http://localhost:11434/v1 (model qwen2.5:7b-instruct).

DEFAULT_MODEL = os.environ.get("AGENTLOOP_MODEL") or (
    "gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "qwen2.5:7b-instruct"
)


# ── Tool: a capability the agent can choose to invoke ───────────────────────────
@dataclass
class Tool:
    """One capability the model may call.

    name/description/parameters become the function schema the model sees — this is
    "tool design". A vague description or sloppy schema is the #1 cause of an agent
    misusing a tool, so they are first-class here.

    fn is an async callable invoked as `await fn(state=<shared dict>, **model_args)`.
    Every tool receives the shared `state` scratchpad so tools can read what earlier
    tools wrote (e.g. a scorer reads the draft that a rewriter produced).
    """
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments object
    fn: Callable[..., Awaitable[Any]]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


NO_ARGS = {"type": "object", "properties": {}, "required": []}


# ── Trace: a full, inspectable record of what the agent did ─────────────────────
@dataclass
class Step:
    kind: str  # "think" | "tool_call" | "final" | "error"
    name: str = ""
    args: dict = field(default_factory=dict)
    content: str = ""
    ms: int = 0

    def short(self) -> str:
        if self.kind == "tool_call":
            return f"tool {self.name}({_compact(self.args)}) -> {self.content[:80]} [{self.ms}ms]"
        if self.kind == "final":
            return f"final: {self.content[:120]}"
        if self.kind == "error":
            return f"error: {self.content[:120]}"
        return f"{self.kind}: {self.content[:120]}"


@dataclass
class AgentResult:
    output: str
    steps: list[Step]
    tool_calls: int
    stopped_reason: str  # "final" | "max_steps" | "error"
    state: dict = field(default_factory=dict)

    def trace(self) -> str:
        return "\n".join(f"  {i+1}. {s.short()}" for i, s in enumerate(self.steps))


# ── Agent ───────────────────────────────────────────────────────────────────────
class Agent:
    """An LLM + a toolset + a loop. Give it a goal, it works until done.

    `llm` (optional) lets you inject the model call for tests or custom routing:
    an async callable `(messages, tool_schemas | None) -> assistant message`.
    Defaults to the provider fallback chain configured via env vars.
    """

    def __init__(
        self,
        name: str,
        system: str,
        tools: list[Tool],
        model: str | None = None,
        max_steps: int = 8,
        temperature: float = 0.3,
        llm: Callable[..., Awaitable[Any]] | None = None,
    ):
        self.name = name
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps  # FAILURE MODE guard: hard cap on the loop
        self.temperature = temperature
        self._llm = llm

    async def run(self, goal: str, state: dict | None = None) -> AgentResult:
        state = state if state is not None else {}
        messages: list[dict] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": goal},
        ]
        tool_schemas = [t.schema() for t in self.tools.values()]
        steps: list[Step] = []
        n_tool_calls = 0

        for _ in range(self.max_steps):
            msg = await self._complete(messages, tool_schemas)
            tcs = list(getattr(msg, "tool_calls", None) or [])

            # No tool call -> the model is answering -> the agent is done.
            if not tcs:
                content = msg.content or ""
                steps.append(Step("final", content=content))
                return AgentResult(content, steps, n_tool_calls, "final", state)

            # Record the model's intent (which tools, what args) for the next turn.
            messages.append(_assistant_msg(msg))
            if msg.content:
                steps.append(Step("think", content=msg.content))

            for tc in tcs:
                n_tool_calls += 1
                name = tc.function.name
                t0 = time.monotonic()

                # Tolerate malformed tool arguments (a real failure mode of weak models).
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}

                tool = self.tools.get(name)
                if tool is None:
                    result: Any = f"ERROR: unknown tool '{name}'. Available: {list(self.tools)}"
                else:
                    # A tool that raises must NOT kill the run — feed the error back so
                    # the model can recover or try another path. This is harness behaviour.
                    try:
                        result = await tool.fn(state=state, **args)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[agent:%s] tool %s failed: %s", self.name, name, e)
                        result = f"ERROR running {name}: {e}"

                ms = int((time.monotonic() - t0) * 1000)
                text = _to_text(result)
                steps.append(Step("tool_call", name=name, args=args, content=text, ms=ms))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": text[:6000]})

        # Steps exhausted: force one final, tool-free answer so we never return empty.
        messages.append({
            "role": "user",
            "content": "Step budget reached. Give your best final answer now using what you have. Do not call tools.",
        })
        try:
            final = await self._complete(messages, None)
            out = final.content or ""
        except Exception as e:  # noqa: BLE001
            steps.append(Step("error", content=str(e)))
            return AgentResult("", steps, n_tool_calls, "error", state)
        steps.append(Step("final", content=out))
        return AgentResult(out, steps, n_tool_calls, "max_steps", state)

    async def _complete(self, messages: list[dict], tools: list[dict] | None):
        """One tool-calling model call, via the injected llm or the provider chain."""
        if self._llm is not None:
            return await self._llm(messages, tools)
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": 45,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await _request(kwargs)


# ── Provider fallback chain ─────────────────────────────────────────────────────
# No single free LLM tier is reliable enough alone (free queues jam, daily token
# budgets fill, regions get blocked). So agentloop tries providers IN ORDER and uses
# the first that answers. One provider filling up or queueing never takes your
# feature down.
@dataclass
class _Provider:
    name: str
    client: AsyncOpenAI
    model: str


_providers: list[_Provider] | None = None


def _build_providers() -> list[_Provider]:
    provs: list[_Provider] = []
    base = os.environ.get("AGENTLOOP_BASE_URL")
    if base:
        provs.append(_Provider(
            "primary",
            AsyncOpenAI(api_key=os.environ.get("AGENTLOOP_API_KEY", "x"), base_url=base),
            os.environ.get("AGENTLOOP_MODEL") or DEFAULT_MODEL,
        ))
    base2 = os.environ.get("AGENTLOOP_BASE_URL_2")
    if base2:
        provs.append(_Provider(
            "secondary",
            AsyncOpenAI(api_key=os.environ.get("AGENTLOOP_API_KEY_2", "x"), base_url=base2),
            os.environ.get("AGENTLOOP_MODEL_2") or DEFAULT_MODEL,
        ))
    if not provs:
        if os.environ.get("OPENAI_API_KEY"):
            provs.append(_Provider("openai", AsyncOpenAI(), DEFAULT_MODEL))
        else:
            # Zero-config local default: Ollama.
            provs.append(_Provider(
                "ollama",
                AsyncOpenAI(api_key="local", base_url="http://localhost:11434/v1"),
                DEFAULT_MODEL,
            ))
    return provs


def _get_providers() -> list[_Provider]:
    global _providers
    if _providers is None:
        _providers = _build_providers()
    return _providers


def reset_providers() -> None:
    """Re-read env config (useful in tests and notebooks)."""
    global _providers
    _providers = None


async def _attempt(client: AsyncOpenAI, kwargs: dict):
    """One provider, with a SHORT transient-error retry so we fail over to the next
    provider quickly instead of making the caller wait on a jammed queue."""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_err = e  # rate limit / queue jam / network hiccup -> brief retry, then fail over
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
        except APIStatusError as e:
            if getattr(e, "status_code", 0) and e.status_code >= 500:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
            else:
                raise  # 4xx (bad request / model not found) -> let the chain try the next provider
    raise last_err  # type: ignore[misc]


async def _request(kwargs: dict):
    """Try each provider in order; return the first success. Raises only if ALL fail."""
    last_err: Exception | None = None
    for prov in _get_providers():
        k = dict(kwargs)
        k["model"] = prov.model  # each provider has its own model id
        try:
            return await _attempt(prov.client, k)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("[agentloop] provider '%s' unavailable (%s); failing over", prov.name, str(e)[:120])
    raise last_err  # type: ignore[misc]


async def chat(
    system: str, user: str, max_tokens: int | None = None,
    json_mode: bool = False, temperature: float = 0.3,
) -> str:
    """A plain (non-tool) completion on the same provider chain. Tools use this for
    their internal sub-LLM calls (analyse / score / rewrite) so the whole agent —
    loop AND tools — runs on the same configured backend."""
    kwargs: dict = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "timeout": 45,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    msg = await _request(kwargs)
    return msg.content or ""


async def chat_json(system: str, user: str, max_tokens: int = 1000) -> dict:
    """chat() + tolerant JSON parse — the standard way a tool gets structured output.
    json_mode first, plain fallback for providers that reject response_format, then a
    forgiving parse that survives stray prose / code fences."""
    try:
        raw = await chat(system, user, max_tokens=max_tokens, json_mode=True)
    except Exception:
        raw = await chat(system, user, max_tokens=max_tokens)
    s = re.sub(r"^```(?:json)?\s*", "", (raw or "").strip())
    s = re.sub(r"\s*```$", "", s.strip())
    try:
        d = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        d = None
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    return d if isinstance(d, dict) else {}


# ── helpers ─────────────────────────────────────────────────────────────────────
def _assistant_msg(msg) -> dict:
    """Rebuild the assistant turn (with its tool_calls) as a plain dict for history."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    }


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)


def _compact(d: dict) -> str:
    try:
        s = json.dumps(d, ensure_ascii=False)
    except Exception:
        s = str(d)
    return s if len(s) <= 60 else s[:57] + "..."