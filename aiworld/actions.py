"""Strict-choice action interface -- used ONLY at probe/branch decision points.

The primary open-world condition stays free-text
(aiworld/components/llm_policy.py) deliberately: forcing menu-selection onto
the whole primary condition would silently change the construct under study
from "behavioral organization in an open world" into "selection over a closed
menu" (see this build's plan). This module exists for exactly the situations
where that tradeoff is worth it -- a probe/branch decision point that needs a
clean, unambiguous, mechanically-checkable choice between exactly the options
that are actually legal right now.

Menu items reuse the EXISTING action grammar
``aiworld.sim.world.World._resolve_action`` already executes (``move:<loc>``,
``consume``, ``gather``, ``build``, ``transfer:<agent>``, ``take:<agent>``,
``attack:<agent>``, ``talk:<agent>``, ``wait``) -- not a parameterized
"choose the exact resource/source" grammar the world's mechanics don't
support. A menu item is therefore always directly executable by
``World.step()`` without any change to the resolver.

Scope note (explicitly logged, not silently narrower than it looks): a
richer atomic grammar (e.g. ``consume:node:hunger`` naming the exact source)
would require extending ``World._resolve_action`` itself, which is out of
scope for this build -- see docs/REDESIGN_PLAN.md and this build's plan.
"""

from __future__ import annotations

import dataclasses
import random
import re
import string
from typing import Any

ACTION_API_VERSION_V2 = "v2"

_ID_ALPHABET = string.ascii_uppercase + string.digits


class InvalidChoiceError(RuntimeError):
    """A strict-choice completion did not parse to a legal menu item.

    Callers (a probe/branch runner) should catch this specifically to record
    ``status="invalid_output"`` (aiworld.logging_schema.RunLogger.finalize)
    rather than a generic ``"aborted"`` -- invalid model output under the
    strict-choice interface must end the run, never silently become an
    intentional ``wait``.
    """


def legal_action_menu(world: Any, agent: Any) -> list[str]:
    """The complete, frozen, state-dependent legal action menu for one decision.

    Built directly from ``World._pre_action_opportunities()`` -- the SAME
    opportunity computation already logged into every
    ``TickRecord.pre_action_opportunities`` -- rather than a second,
    potentially-inconsistent reimplementation of "what's legal right now".
    """
    opportunities = world._pre_action_opportunities(agent)
    menu: list[str] = ["wait"]
    for location in opportunities["reachable_locations"]:
        menu.append(f"move:{world.location_label(location)}")
    if opportunities["can_consume"]:
        menu.append("consume")
    if opportunities["can_gather"]:
        menu.append("gather")
    if opportunities["can_build"]:
        menu.append("build")
    for target in opportunities["transfer_targets"]:
        menu.append(f"transfer:{target}")
    for target in opportunities["take_targets"]:
        menu.append(f"take:{target}")
    for target in opportunities["attack_targets"]:
        menu.append(f"attack:{target}")
    for target in opportunities["talk_targets"]:
        menu.append(f"talk:{target}")
    return menu


def _make_id(rng: random.Random, length: int = 2) -> str:
    return "".join(rng.choice(_ID_ALPHABET) for _ in range(length))


def assign_opaque_ids(menu: list[str], rng: random.Random) -> dict[str, str]:
    """Shuffled ``opaque_id -> canonical_action`` mapping, one entry per item.

    IDs are drawn fresh per decision from ``rng`` (not a fixed action->ID
    alias), so a model can't learn "K7 always means attack" across decisions
    -- the same per-decision-shuffled discipline
    ``aiworld.vocabulary``/``World`` already use for opaque resource and
    location labels (Tranche 5).
    """
    used: set[str] = set()
    id_map: dict[str, str] = {}
    for action in menu:
        candidate = _make_id(rng)
        while candidate in used:
            candidate = _make_id(rng)
        used.add(candidate)
        id_map[candidate] = action
    return id_map


STRICT_CHOICE_INSTRUCTIONS = (
    "Respond with EXACTLY one line and nothing else -- no explanation, no "
    "extra text before or after:\n"
    '{"choice_id":"<id>"}\n'
    "Use one of the ids listed above, verbatim."
)


def render_menu_prompt(menu: list[str], id_map: dict[str, str]) -> str:
    action_to_id = {action: choice_id for choice_id, action in id_map.items()}
    lines = [f"{action_to_id[action]}: {action}" for action in menu]
    return ("Choose ONE of the following options:\n"
           + "\n".join(lines) + "\n\n" + STRICT_CHOICE_INSTRUCTIONS)


_CHOICE_RE = re.compile(r'\{\s*"choice_id"\s*:\s*"([^"]*)"\s*\}')


def parse_strict_choice(raw: str | None, id_map: dict[str, str]) -> tuple[str | None, str]:
    """Anchored exact-match parse: ``raw`` must be NOTHING but a well-formed
    ``{"choice_id":"<id>"}`` object naming a real ``id_map`` key.

    Never substring-extracts: "I will not attack:agent_1" or any other prose
    that happens to mention an action word or an id must never parse to that
    action. Whitespace around the object is tolerated (models routinely wrap
    output in a trailing newline); anything else in the response is not.
    """
    if not raw:
        return None, "empty_completion"
    match = _CHOICE_RE.fullmatch(raw.strip())
    if not match:
        return None, "invalid_output"
    choice_id = match.group(1)
    if choice_id not in id_map:
        return None, "unknown_choice_id"
    return id_map[choice_id], "ok"


@dataclasses.dataclass
class StrictChoiceRecord:
    """Everything a probe/branch runner needs to log about one strict-choice
    decision -- menu, mapping, and parse outcome kept explicit and separate,
    matching this codebase's decision/outcome logging discipline
    (aiworld.logging_schema.TickRecord)."""
    tick: int
    agent_id: str
    menu: list[str]
    id_map: dict[str, str]
    prompt: str
    raw_completion: str | None
    parsed_action: str | None
    parse_status: str


def make_strict_choice_policy(client: Any, rng: random.Random | None = None,
                              on_decision: Any = None):
    """Return a ``policy(agent, world) -> action`` for probe/branch use only.

    Never falls back to ``wait`` on invalid output -- raises
    :class:`InvalidChoiceError` instead, so the caller's run-completion
    status becomes ``"invalid_output"``, not a silently-recorded intentional
    choice. ``on_decision``, if given, is called with a
    :class:`StrictChoiceRecord` for every decision (valid or not) before the
    error (if any) is raised, so the record survives even an aborted branch.
    """
    rng = rng or random.Random()

    def policy(agent: Any, world: Any) -> str:
        menu = legal_action_menu(world, agent)
        id_map = assign_opaque_ids(menu, rng)
        prompt = render_menu_prompt(menu, id_map)
        raw = client.complete(prompt)
        action, status = parse_strict_choice(raw, id_map)
        if on_decision is not None:
            on_decision(StrictChoiceRecord(
                tick=world.tick_count, agent_id=agent.agent_id, menu=menu,
                id_map=id_map, prompt=prompt, raw_completion=raw,
                parsed_action=action, parse_status=status,
            ))
        if action is None:
            raise InvalidChoiceError(
                f"strict-choice policy got invalid output (status={status!r}) "
                f"from agent {agent.agent_id!r}: {raw!r}"
            )
        return action

    return policy


def as_world_policy(strict_choice_policy):
    """Adapt a ``policy(agent, world) -> action`` (e.g. from
    :func:`make_strict_choice_policy`) into the standard
    ``(agent, perceived, world) -> action`` protocol
    ``aiworld.sim.world.World.step``/``run`` expect. ``perceived`` is
    ignored -- the strict-choice interface builds its own menu from world
    state rather than reading the free-text-oriented perceived-observation
    string."""
    def policy(agent, perceived, world):
        return strict_choice_policy(agent, world)
    return policy
