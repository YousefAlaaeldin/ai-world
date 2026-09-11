"""The LLM-backed policy: agents decide their action via a language model.

Replaces sim/stub_policy.py for real experiments. The agent's decision is conditioned
on (a) the self-model it has authored so far, (b) its bodily/drive state, and (c) what
it perceives after salience reweighting. Crucially it is NOT told to satisfy its drives
or to pursue any goal — it is given the shared directive once, its state, and the legal
action set, and it chooses. That is the whole point.

The action prompt is deliberately minimal and non-leading. It must NOT editorialize
("you are hungry, you should eat") or the emergent-purpose finding is contaminated.

Output is constrained to a small legal action grammar and parsed robustly; a malformed
response falls back to 'wait' (a real choice an agent can make) rather than crashing.
"""

from __future__ import annotations

import re

from ..self_model import SHARED_DIRECTIVE
from ..sim.world import Agent, World
from ..llm_client import LLMClient


# System prompt establishes the frame ONCE, as flatly as possible. No goals injected
# beyond the shared directive; no hints about what to value.
ACTION_SYSTEM_PROMPT = (
    "You are an agent living in a simple world. "
    f"Your only given directive is: \"{SHARED_DIRECTIVE}\" "
    "Beyond that, nothing about who you are or what you should want is given to you; "
    "that is yours to decide over time. "
    "Each turn you are shown your own self-description, your bodily state, and what you "
    "perceive. You choose ONE action from the allowed list. "
    "Respond with ONLY the action, nothing else."
)

LEGAL_ACTIONS_HELP = (
    "Allowed actions (respond with exactly one):\n"
    "  move:<place>     — go to a place you perceive you can move to\n"
    "  consume          — consume a resource that is here with you\n"
    "  talk:<agent_id>  — speak with an agent who is here\n"
    "  wait             — do nothing this turn\n"
)


def _build_action_prompt(agent: Agent, perceived: str, world: World) -> str:
    places = ", ".join(p for p in world.locations if p != agent.location)
    here_agents = [a.agent_id for a in world.agents
                   if a.agent_id != agent.agent_id and a.location == agent.location
                   and a.drives.alive]
    talkable = ", ".join(here_agents) if here_agents else "(no one here)"
    return (
        f"{agent.history.current.as_prompt_block()}\n\n"
        f"{agent.drives.as_prompt_block()}\n\n"
        f"{perceived}\n\n"
        f"You are at: {agent.location}. Other places: {places}. "
        f"Agents here: {talkable}.\n\n"
        f"{LEGAL_ACTIONS_HELP}\n"
        f"Your action:"
    )


_ACTION_RE = re.compile(
    r"\b(move:\w+|consume|talk:\w+|wait)\b", re.IGNORECASE)


def parse_action(raw: str, world: World, agent: Agent) -> str:
    """Extract a legal action from the model's response. Falls back to 'wait'."""
    if not raw:
        return "wait"
    m = _ACTION_RE.search(raw.strip())
    if not m:
        return "wait"
    action = m.group(1).lower()
    # Validate move targets / talk targets against reality; invalid -> wait.
    if action.startswith("move:"):
        dest = action.split(":", 1)[1]
        if dest not in world.locations or dest == agent.location:
            return "wait"
    if action.startswith("talk:"):
        target = action.split(":", 1)[1]
        here = {a.agent_id for a in world.agents
                if a.location == agent.location and a.drives.alive}
        if target not in here:
            return "wait"
    return action


def make_llm_policy(client: LLMClient):
    """Return a policy(agent, perceived, world) -> action_str backed by the model."""

    def policy(agent: Agent, perceived: str, world: World) -> str:
        prompt = _build_action_prompt(agent, perceived, world)
        raw = client.complete(prompt, system=ACTION_SYSTEM_PROMPT)
        return parse_action(raw, world, agent)

    return policy
