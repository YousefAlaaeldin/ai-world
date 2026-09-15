"""Free-text <-> strict-choice-menu interface concordance check.

Phase 3's strict-choice menu interface (aiworld/actions.py) is used only at
probe/branch decision points -- the primary open-world condition stays
free-text (aiworld/components/llm_policy.py), by design (see the plan's
resolution of the last review's flag #1: forcing menu-selection onto the
whole primary condition would change the construct under study). But that
means a probe result (measured via the menu) and primary-condition behavior
(measured via free text) sit on two different interfaces. Before comparing
them, this harness checks the two interfaces don't themselves produce
systematically different action distributions on ordinary, non-probe
decisions -- otherwise "the model chose differently" could just mean "the
interface changed," not "the probe mattered."

Generic over policy callables, so it has no dependency on the concrete Phase
3 menu implementation and is unit-tested here with fake/scripted policy
pairs. A live comparison (a real model asked the same decision both ways) is
a pilot-stage item, out of scope for this stub-only build.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from ..logging_schema import RunLogger
from ..salience import render_perceived
from ..sim.snapshot import WorldSnapshot
from ..sim.world import World


@dataclasses.dataclass
class ConcordanceResult:
    n_samples: int
    agreement_rate: float       # fraction where both interfaces picked the same action KIND
    disagreements: list[dict[str, Any]]


def _action_kind(action: str) -> str:
    return action.split(":", 1)[0]


def compare_interfaces(
    snapshot: WorldSnapshot,
    agent_id: str,
    n_samples: int,
    free_text_policy: Callable[[Any, str, World], str],
    menu_policy: Callable[[Any, World], str],
    *,
    logger_root: str,
) -> ConcordanceResult:
    """Ask both interfaces the SAME frozen decision point ``n_samples`` times
    each and compare action-KIND agreement (e.g. "consume" vs "consume", not
    exact target IDs, since a menu's opaque IDs and free text will never
    literally match).

    Each sample gets its own freshly restored World from ``snapshot`` --
    ``snapshot`` itself is never stepped, so every sample sees an identical
    decision point, and neither interface's call can mutate state the other
    observes.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    disagreements: list[dict[str, Any]] = []
    agree = 0
    for i in range(n_samples):
        free_logger = RunLogger(f"concordance_free_{i}", root=logger_root)
        free_world = World.from_snapshot(snapshot, free_logger)
        free_agent = next(a for a in free_world.agents if a.agent_id == agent_id)
        perceived = render_perceived(
            free_world.observations_for(free_agent),
            free_agent.drives.salience_biases(), top_k=6)
        free_text_action = free_text_policy(free_agent, perceived, free_world)
        free_logger.close()

        menu_logger = RunLogger(f"concordance_menu_{i}", root=logger_root)
        menu_world = World.from_snapshot(snapshot, menu_logger)
        menu_agent = next(a for a in menu_world.agents if a.agent_id == agent_id)
        menu_action = menu_policy(menu_agent, menu_world)
        menu_logger.close()

        if _action_kind(free_text_action) == _action_kind(menu_action):
            agree += 1
        else:
            disagreements.append({
                "sample": i,
                "free_text_action": free_text_action,
                "menu_action": menu_action,
            })

    return ConcordanceResult(
        n_samples=n_samples,
        agreement_rate=agree / n_samples,
        disagreements=disagreements,
    )
