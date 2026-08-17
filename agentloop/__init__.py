"""agentloop — a minimal, transparent LLM agent framework with a built-in eval harness."""
from .core import (
    NO_ARGS,
    Agent,
    AgentResult,
    Step,
    Tool,
    chat,
    chat_json,
    reset_providers,
)
from .evals import Report, Scenario, ScenarioResult, markdown_report, print_report, run, run_evals

__version__ = "0.1.1"

__all__ = [
    "Agent",
    "AgentResult",
    "Step",
    "Tool",
    "NO_ARGS",
    "chat",
    "chat_json",
    "reset_providers",
    "Scenario",
    "ScenarioResult",
    "Report",
    "run",
    "run_evals",
    "print_report",
    "markdown_report",
    "__version__",
]