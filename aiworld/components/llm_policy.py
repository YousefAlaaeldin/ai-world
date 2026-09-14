"""The LLM-backed policy: agents decide their action via a language model.

Replaces sim/stub_policy.py for real experiments.  Which persistent context an
action receives is an explicit experimental treatment: none, a self-authored
self-model, or an objective factual ledger.  The open-world conditions do not give
the model an assigned distal goal; the historical "survive and find your purpose"
prompt remains available only as an explicit positive control.

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
from ..logging_schema import CompletionRecord


ACTION_MEMORY_NONE = "none"
ACTION_MEMORY_FACTUAL_LEDGER = "factual_ledger"
ACTION_MEMORY_MODES = frozenset({
    ACTION_MEMORY_NONE,
    ACTION_MEMORY_FACTUAL_LEDGER,
})
DEFAULT_ACTION_MEMORY_MAX_EVENTS = 12
DEFAULT_ACTION_CONTEXT_MAX_CHARS = 1_200


def _validate_action_memory_mode(mode: str) -> None:
    if mode not in ACTION_MEMORY_MODES:
        choices = ", ".join(sorted(ACTION_MEMORY_MODES))
        raise ValueError(
            f"Unknown action memory mode {mode!r}; expected one of: {choices}."
        )


def _validate_context_bounds(max_events: int, max_chars: int) -> None:
    if max_events < 1:
        raise ValueError("action memory max_events must be at least 1.")
    # The header plus one useful event need room.  This guard also prevents a
    # config typo from turning the factual-memory treatment into a hidden
    # no-memory arm.
    if max_chars < 96:
        raise ValueError("action context max_chars must be at least 96.")


def _event_tick(entry: str) -> int:
    """Extract a ledger tick for stable ordering without trusting prose."""
    match = re.match(r"\s*tick\s+(\d+)\s*:", entry, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _factual_ledger(agent: Agent, *, max_events: int,
                    max_chars: int) -> str:
    """Return a bounded, non-draining event ledger for action selection.

    The factual-memory condition must give the model history without reusing
    ``agent_recent_memories()``, whose "you remember" framing and draining
    behavior were designed for a self-model reflection prompt.  The world writes
    factual event strings when ``memory_style: factual``; the runner rejects the
    unsafe narrative/ledger combination before a real run begins.

    We keep only the newest complete entries that fit in the declared bound.  The
    cap is part of the condition and gets persisted in run metadata, so later
    analyses can distinguish a memory treatment from accidental context growth.
    """
    entries = [str(entry).strip() for entry in (
        [*agent.long_term_memory, *agent.recent_memories]
    ) if str(entry).strip()]
    # Long-term and recent buffers can interleave.  Sorting by the explicit tick
    # prefix produces a readable objective chronology while retaining stable
    # insertion order for entries sharing a tick.
    entries = [entry for _, entry in sorted(
        enumerate(entries), key=lambda item: (_event_tick(item[1]), item[0]))]
    entries = entries[-max_events:]

    header = "Factual event ledger (oldest to newest):"
    lines: list[str] = []
    used = len(header)
    for entry in reversed(entries):
        line = f"- {entry}"
        additional = len(line) + 1
        if lines and used + additional > max_chars:
            break
        if not lines and used + additional > max_chars:
            # Keep an explicitly bounded (rather than silently unbounded)
            # representation even if one malformed world event is huge.
            # Account for the joining newline as well as "- " and the ellipsis.
            available = max(1, max_chars - used - 1 - len("- ") - len("…"))
            lines.append(f"- {entry[-available:]}…")
            break
        lines.append(line)
        used += additional
    if not lines:
        return f"{header}\n- (no recorded events)"
    lines.reverse()
    return "\n".join([header, *lines])


def _bounded_context(block: str, max_chars: int) -> str:
    """Bound dynamic treatment context in every arm that receives one."""
    if len(block) <= max_chars:
        return block
    return f"{block[:max_chars - 1]}…"


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
    state_word = ("bodily state" if vocabulary == "biological"
                  else "state" if vocabulary == "opaque" else "internal state")
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


def _build_action_prompt(
    agent: Agent,
    perceived: str,
    world: World,
    show_self_model: bool = True,
    action_memory: str = ACTION_MEMORY_NONE,
    action_memory_max_events: int = DEFAULT_ACTION_MEMORY_MAX_EVENTS,
    action_context_max_chars: int = DEFAULT_ACTION_CONTEXT_MAX_CHARS,
) -> str:
    """Build one action prompt with an explicitly assigned context treatment.

    ``show_self_model`` and ``action_memory`` are deliberately separate.  A
    factual ledger supplies observed event history, not a self-description; the
    condition cannot silently gain identity language by turning memory on.
    """
    _validate_action_memory_mode(action_memory)
    _validate_context_bounds(action_memory_max_events, action_context_max_chars)
    # Only places reachable in one move -- telling the model it can go anywhere
    # when the world only lets it reach neighbors would make every 'wait'
    # fallback (an unreachable move gets silently demoted to wait) look like a
    # model failure instead of what it is, a world-topology constraint.
    reachable = ", ".join(
        sorted(world.location_label(loc)
               for loc in world.adjacency.get(agent.location, ())))
    places = reachable or "(nowhere reachable from here)"
    here_agents = [a.agent_id for a in world.agents
                   if a.agent_id != agent.agent_id and a.location == agent.location
                   and a.drives.alive]
    talkable = ", ".join(here_agents) if here_agents else "(no one here)"
    # In the de-primed baseline the self-model is NOT shown back to the agent, so
    # its own prior self-description can't steer the next action.  The factual
    # ledger is a distinct treatment: it contains no identity fields and is
    # bounded by the same declared dynamic-context cap.
    self_block = (
        f"{_bounded_context(agent.history.current.as_prompt_block(), action_context_max_chars)}\n\n"
        if show_self_model else ""
    )
    ledger_block = (
        f"{_factual_ledger(agent, max_events=action_memory_max_events, max_chars=action_context_max_chars)}\n\n"
        if action_memory == ACTION_MEMORY_FACTUAL_LEDGER else ""
    )
    # What the agent is carrying, vocabulary-routed so the neutral arm doesn't leak.
    held = {k: v for k, v in agent.inventory.items() if v > 0}
    carried = (", ".join(f"{world.resource_noun(k)} x{v}"
                         for k, v in held.items()) if held else "nothing")
    return (
        f"{self_block}"
        f"{ledger_block}"
        f"{agent.drives.as_prompt_block()}\n\n"
        f"You are carrying: {carried}.\n\n"
        f"{perceived}\n\n"
        f"You are at: {world.location_label(agent.location)}. "
        f"Places you could move to: {places}. "
        f"Agents here: {talkable}.\n\n"
        f"{LEGAL_ACTIONS_HELP}\n"
        f"Your action:"
    )


_ACTION_RE = re.compile(
    r"\b(move:\w+|transfer:\w+|take:\w+|attack:\w+|talk:\w+"
    r"|consume|gather|build|wait)\b", re.IGNORECASE)


def parse_action_diagnostic(raw: str, world: World, agent: Agent) -> tuple[str, str]:
    """Extract an action plus an auditable parse status.

    A raw ``wait`` is a valid decision, whereas a malformed response falling
    back to ``wait`` is a data-quality event.  The distinction is required for
    reproducible behavioral analysis and is intentionally kept out of the
    action grammar itself.
    """
    if not raw:
        return "wait", "empty_completion"
    m = _ACTION_RE.search(raw.strip())
    if not m:
        return "wait", "no_legal_action"
    action = m.group(1).lower()
    # Validate targets against reality; invalid -> wait.
    if action.startswith("move:"):
        supplied = action.split(":", 1)[1]
        # In an opaque-location condition, accept only the displayed label from
        # model output, then convert it back to the internal resolver key.
        dest = world.canonical_location(
            supplied, allow_internal=world.vocabulary != "opaque")
        if dest is None or dest not in world.adjacency.get(agent.location, ()):
            return "wait", "invalid_move"
        action = f"move:{dest}"
    # Every directed action needs a living, co-located target.
    if action.split(":", 1)[0] in ("talk", "transfer", "take", "attack") and ":" in action:
        target = action.split(":", 1)[1]
        if world._here_target(agent, target) is None:
            return "wait", "invalid_target"
    return action, "ok"


def parse_action(raw: str, world: World, agent: Agent) -> str:
    """Backward-compatible action-only wrapper around the diagnostic parser."""
    return parse_action_diagnostic(raw, world, agent)[0]


def make_llm_policy(client: LLMClient, vocabulary: str = "neutral",
                    directive: str | None = SHARED_DIRECTIVE,
                    show_self_model: bool = True,
                    action_memory: str = ACTION_MEMORY_NONE,
                    action_memory_max_events: int = DEFAULT_ACTION_MEMORY_MAX_EVENTS,
                    action_context_max_chars: int = DEFAULT_ACTION_CONTEXT_MAX_CHARS):
    """Return a policy(agent, perceived, world) -> action_str backed by the model.

    Condition knobs (docs/REDESIGN_PLAN.md):
      vocabulary       'neutral' | 'biological' framing (aiworld/vocabulary.py)
      directive        assigned goal text, or None/"" for the de-primed baseline
      show_self_model  whether the self-model is fed back into the action prompt
      action_memory    'none' | 'factual_ledger'; observed history, never identity
      action_memory_max_events / action_context_max_chars
                       declared bounds on persistent action context
    Defaults reproduce the original primed arm.
    """
    _validate_action_memory_mode(action_memory)
    _validate_context_bounds(action_memory_max_events, action_context_max_chars)
    system_prompt = _action_system_prompt(vocabulary, directive, show_self_model)

    def policy(agent: Agent, perceived: str, world: World) -> str:
        prompt = _build_action_prompt(agent, perceived, world,
                                      show_self_model=show_self_model,
                                      action_memory=action_memory,
                                      action_memory_max_events=action_memory_max_events,
                                      action_context_max_chars=action_context_max_chars)
        raw = client.complete(prompt, system=system_prompt)
        action, status = parse_action_diagnostic(raw, world, agent)
        # World.run emits the structural per-tick trace.  Retaining the full
        # request/response on the agent lets this policy work with a tiny fake
        # client in tests as well as with the logging-aware real client.
        agent.last_action_prompt = prompt
        agent.last_action_system_prompt = system_prompt
        agent.last_raw_completion = raw
        agent.last_parse_status = status
        request_metadata = {}
        response_metadata = {}
        get_request_metadata = getattr(client, "request_metadata", None)
        get_response_metadata = getattr(client, "response_metadata", None)
        if callable(get_request_metadata):
            request_metadata = get_request_metadata()
        if callable(get_response_metadata):
            response_metadata = get_response_metadata()
        world.logger.log_completion(CompletionRecord(
            phase="action",
            tick=world.tick_count,
            agent_id=agent.agent_id,
            system_prompt=system_prompt,
            user_prompt=prompt,
            raw_completion=raw,
            parsed_output=action,
            parse_status=status,
            request_metadata=request_metadata,
            response_metadata=response_metadata,
        ))
        return action

    return policy
