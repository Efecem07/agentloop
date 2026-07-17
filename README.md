# agentloop

A minimal, transparent LLM agent framework with a built-in eval harness. No magic:
the entire agent loop is one readable file, and "does the agent actually work?" is
answered by scenario-based evals instead of vibes.

Extracted from the agent runtime I built for [MatchupWise](https://matchupwise.com),
where it powers CV-tailoring, career-path analysis and interview-prep agents in
production.

## Why another agent framework?

It isn't one, really. Popular frameworks hide exactly the parts you need to
understand when an agent misbehaves: what the model saw, which tool it chose, what
came back, why the run stopped. agentloop keeps all of that on one screen and makes
every run fully inspectable:

- **`Agent`** = an LLM + a toolset + a loop (ReAct: reason, act, observe, repeat).
- **`Tool`** = name + description + JSON schema + an async function. Tool design is
  first-class because a vague description is the #1 cause of tool misuse.
- **Failure modes are handled, not hidden**: `max_steps` caps the loop, a raising
  tool becomes an observation the model can react to, malformed tool arguments are
  tolerated, transient provider errors retried, and when the step budget runs out
  the agent is forced to produce a final answer instead of returning nothing.
- **Every run is a trace**: each thought and tool call is recorded as a `Step`, so
  runs can be inspected, replayed and graded.
- **Provider fallback chain**: point it at any OpenAI-compatible endpoint; give it
  two and it fails over automatically (free-tier queues jam, budgets fill — one
  provider being down never takes the feature down).

## Evals as a first-class citizen

Non-deterministic systems need test suites too. `agentloop.evals` runs an agent
over scenarios and grades each run with explicit, deterministic checks that return
*reasons*, not booleans:

```python
from agentloop import Scenario, run

def grade(res: dict) -> list[str]:
    fails = []
    if len(res.get("score_history") or []) < 2:
        fails.append("did not iterate")
    if res.get("final_score", 0) < 70:
        fails.append(f"final score {res.get('final_score')} < 70")
    return fails                       # [] = pass

run([Scenario("buzzword_draft", lambda: polish(), grade)], repeat=3)
```

`repeat=N` reruns every scenario and fails it if *any* repeat fails — a scenario
that passes 1 time in 3 is a bug, not a pass.

### Current results

Bundled example agents, graded end-to-end against a real model (`gpt-4o-mini`,
`--repeat 3`, so each scenario ran 3 times and had to pass every time):

| scenario | result | details |
|---|---|---|
| log_detective/incident | ✅ pass | tool_calls=4, stopped_reason=final |
| copy_polisher/buzzword_draft | ✅ pass | score_history=[40, 85], final_score=85 |
| copy_polisher/vague_draft | ✅ pass | score_history=[30, 85], final_score=85 |

The score histories are the point: the polisher scored a baseline, rewrote, and
re-scored — the eval verifies the agent *iterated with feedback*, not that it got
lucky once. Reproduce with `python -m examples.evals --repeat 3 --markdown`.

### What a run looks like

A real `log_detective` trace (deterministic tools, model decisions):

```text
tool count_by_status({}) -> {"200": 8, "500": 6}
tool search_log({"pattern": "500"}) -> {"matches": ["...ERROR POST /api/checkout 500 2004ms db_pool_ex...
tool tail_log({"n": 20}) -> {"lines": ["2026-07-14T09:58:01 INFO  GET /api/products 200 34ms", ...
tool report({"failing_endpoint": "/api/checkout", "root_cause": "db_p...) -> {...}
final: The /api/checkout endpoint is failing due to a "db_pool_exhausted" issue,
       which started at 2026-07-14T10:01:02.
```

It counted statuses to confirm something is wrong, grepped the 500s to find the
endpoint and cause, checked what preceded them, then filed a structured report.

## Quickstart

```bash
pip install -e .

# any OpenAI-compatible endpoint works:
export AGENTLOOP_BASE_URL=https://api.groq.com/openai/v1
export AGENTLOOP_API_KEY=gsk_...
export AGENTLOOP_MODEL=llama-3.3-70b-versatile

# or zero-config with a local Ollama (no key needed):
#   ollama pull qwen2.5:7b-instruct

python -m examples.log_detective   # deterministic tools, watch the agent investigate
python -m examples.copy_polisher   # iterate-until-score loop
python -m examples.evals           # grade both agents end-to-end
```

Optional fallback provider: set `AGENTLOOP_BASE_URL_2` / `AGENTLOOP_API_KEY_2` /
`AGENTLOOP_MODEL_2` and the chain fails over automatically.

## Writing an agent

```python
from agentloop import Agent, Tool, NO_ARGS

async def count_by_status(state: dict, **_) -> dict:
    ...                                # plain Python; gets the shared state dict

async def search_log(state: dict, pattern: str = "", **_) -> dict:
    ...

agent = Agent(
    name="log_detective",
    system="You are an incident-response agent. Work with tools, do not guess...",
    tools=[
        Tool("count_by_status", "Count log lines per HTTP status.", NO_ARGS, count_by_status),
        Tool("search_log", "Return log lines matching a regex.",
             {"type": "object", "properties": {"pattern": {"type": "string"}},
              "required": ["pattern"]}, search_log),
    ],
    max_steps=8,
)

result = await agent.run("Something is wrong in production. Investigate.", state={"log": log_text})
print(result.output)          # the diagnosis
print(result.trace())         # every thought + tool call, with timings
```

Tools share a `state` dict (a scratchpad): a scorer can read the draft a rewriter
produced, and your code can read everything after the run. The two bundled examples
show the two most useful shapes:

- [`examples/log_detective.py`](examples/log_detective.py) — deterministic tools
  (grep/count/slice over a log); only the *decisions* are the model's.
- [`examples/copy_polisher.py`](examples/copy_polisher.py) — the iterate-until-score
  pattern: a strict LLM judge scores the text, the agent rewrites and re-scores until
  the score stops improving. The same pattern powers CV tailoring in production.

## Testing without a network

`Agent(llm=...)` injects the model call, so harness behaviour is unit-testable with
a scripted fake — see [`tests/test_core.py`](tests/test_core.py) for coverage of the
failure modes (unknown tool, raising tool, malformed args, max-steps cutoff).

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT