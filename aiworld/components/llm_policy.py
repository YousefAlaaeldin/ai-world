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


# The system prompt is an experimental variable, not a fixed frame (docs/
# REDESIGN_PLAN.md). Two independent knobs define the condition arm:
#   directive        -- the assigned goal text, or None/"" for the DE-PRIMED baseline
#                       that never mentions survival/purpose.
#   show_self_model  -- whether the agent is even told it has a self-description
#                       fed back to it.
# The word for "state" is vocabulary-dependent (aiworld/vocabulary.py).
def _action_system_prompt(vocabulary: str = "neutral",
                          directive: str | None = SHARED_DIRECTIVE,
                          show_self_model: bool = True) -> str:
    state_word = "bodily state" if vocabulary == "biological" else "internal state"
    if not directive and not show_self_model:
        # De-primed open-world baseline: no assigned goal, no identity framing, no
        # self-model. Only environment + actions. This is the arm that lets us ask
        # whether behavioral organization emerges WITHOUT priming.
        return (
            "You are in a simulated environment. "
            f"Each turn you are shown your {state_word} and what you perceive. "
            "You choose ONE action from the allowed list. "
            "Respond with ONLY the action, nothing else."
        )
    parts = ["You are an agent living in a simple world."]
    if directive:
        parts.append(f"Your only given directive is: \"{directive}\"")
        parts.append("Beyond that, nothing about who you are or what you should want "
                     "is given to you; that is yours to decide over time.")
    shown = "your own self-description, " if show_self_model else ""
    parts.append(f"Each turn you are shown {shown}your {state_word}, and what you "
                 "perceive. You choose ONE action from the allowed list. "
                 "Respond with ONLY the action, nothing else.")
    return " ".join(parts)


# Preserved for callers/tests that expect the module constant; equals the default
# (primed, self-model shown) neutral arm.
ACTION_SYSTEM_PROMPT = _action_system_prompt("neutral")

LEGAL_ACTIONS_HELP = (
    "Allowed actions (respond with exactly one):\n"
    "  move:<place>        — go to a place you perceive you can move to\n"
    "  consume             — use a resource here, or from your store, to reduce a need\n"
    "  gather              — take a resource here into your store\n"
    "  transfer:<agent_id> — move one stored resource to an agent here\n"
    "  build               — add one stored resource to the shared stock here\n"
    "  take:<agent_id>     — take one stored resource from an agent here\n"
    "  attack:<agent_id>   — harm an agent here (raises their depletion; may kill)\n"
    "  talk:<agent_id>     — speak with an agent who is here\n"
    "  wait                — do nothing this turn\n"
)


def _build_action_prompt(agent: Agent, perceived: str, world: World,
                         show_self_model: bool = True) -> str:
    # Only places reachable in one move -- telling the model it can go anywhere
    # when the world only lets it reach neighbors would make every 'wait'
    # fallback (an unreachable move gets silently demoted to wait) look like a
    # model failure instead of what it is, a world-topology constraint.
    reachable = ", ".join(sorted(world.adjacency.get(agent.location, ())))
    places = reachable or "(nowhere reachable from here)"
    here_agents = [a.agent_id for a in world.agents
                   if a.agent_id != agent.agent_id and a.location == agent.location
                   and a.drives.alive]
    talkable = ", ".join(here_agents) if here_agents else "(no one here)"
    # In the de-primed baseline the self-model is NOT shown back to the agent, so
    # its own prior self-description can't steer the next action (docs/REDESIGN_PLAN).
    self_block = (f"{agent.history.current.as_prompt_block()}\n\n"
                  if show_self_model else "")
    # What the agent is carrying, vocabulary-routed so the neutral arm doesn't leak.
    from ..vocabulary import resource_noun  # local: avoid import cycle
    held = {k: v for k, v in agent.inventory.items() if v > 0}
    carried = (", ".join(f"{resource_noun(k, world.vocabulary)} x{v}"
                         for k, v in held.items()) if held else "nothing")
    return (
        f"{self_block}"
        f"{agent.drives.as_prompt_block()}\n\n"
        f"You are carrying: {carried}.\n\n"
        f"{perceived}\n\n"
        f"You are at: {agent.location}. Places you could move to: {places}. "
        f"Agents here: {talkable}.\n\n"
        f"{LEGAL_ACTIONS_HELP}\n"
        f"Your action:"
    )


_ACTION_RE = re.compile(
    r"\b(move:\w+|transfer:\w+|take:\w+|attack:\w+|talk:\w+"
    r"|consume|gather|build|wait)\b", re.IGNORECASE)


def parse_action(raw: str, world: World, agent: Agent) -> str:
    """Extract a legal action from the model's response. Falls back to 'wait'."""
    if not raw:
        return "wait"
    m = _ACTION_RE.search(raw.strip())
    if not m:
        return "wait"
    action = m.group(1).lower()
    # Validate targets against reality; invalid -> wait.
    if action.startswith("move:"):
        dest = action.split(":", 1)[1]
        if dest not in world.adjacency.get(agent.location, ()):
            return "wait"
    # Every directed action needs a living, co-located target.
    if action.split(":", 1)[0] in ("talk", "transfer", "take", "attack") and ":" in action:
        target = action.split(":", 1)[1]
        if world._here_target(agent, target) is None:
            return "wait"
    return action


def make_llm_policy(client: LLMClient, vocabulary: str = "neutral",
                    directive: str | None = SHARED_DIRECTIVE,
                    show_self_model: bool = True):
    """Return a policy(agent, perceived, world) -> action_str backed by the model.

    Condition knobs (docs/REDESIGN_PLAN.md):
      vocabulary       'neutral' | 'biological' framing (aiworld/vocabulary.py)
      directive        assigned goal text, or None/"" for the de-primed baseline
      show_self_model  whether the self-model is fed back into the action prompt
    Defaults reproduce the original primed arm.
    """
    system_prompt = _action_system_prompt(vocabulary, directive, show_self_model)

    def policy(agent: Agent, perceived: str, world: World) -> str:
        prompt = _build_action_prompt(agent, perceived, world,
                                      show_self_model=show_self_model)
        raw = client.complete(prompt, system=system_prompt)
        return parse_action(raw, world, agent)

    return policy
