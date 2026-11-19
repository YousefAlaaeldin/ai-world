"""communication_mode='informational': talk can pass one objective resource-
location fact the speaker personally verified -- never identity content,
never freely generated text (that distinction is the whole point: 'none' and
'self_model' already existed; this is deliberately a third, narrower channel).

Built directly off the Phase 0.5 pilot (2026-09-15): agents that never
personally visited a resource had no way to learn about it short of stumbling
onto it themselves. This tests whether letting a resource-savvy agent tell a
stuck one changes anything -- not just implements the mechanism.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402


def _agent(agent_id: str, location: str) -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def _world(tmp_path, communication_mode="informational"):
    logger = RunLogger("talk_test", root=str(tmp_path))
    agents = [_agent("a0", "hub"), _agent("a1", "hub")]
    resources = [ResourceNode("east", DriveKind.HUNGER, quantity=5)]
    adjacency = {"hub": {"east"}, "east": {"hub"}}
    world = World(["hub", "east"], resources, agents, logger,
                 adjacency=adjacency, communication_mode=communication_mode,
                 memory_style="factual")
    return world, logger


def test_informational_mode_is_accepted_alongside_existing_modes(tmp_path):
    world, logger = _world(tmp_path, communication_mode="informational")
    logger.close()
    assert world.communication_mode == "informational"


def test_talk_with_nothing_to_share_stays_content_free(tmp_path):
    world, logger = _world(tmp_path)
    a0, a1 = world.agents
    outcome = world.resolve_action(a0, "talk:a1")
    logger.close()
    assert "nothing to share" in outcome
    assert any("nothing to share" in m for m in a1.recent_memories)
    # Affiliation is still satisfied even with no information payload.
    assert a0.drives.states[DriveKind.AFFILIATION].level == 0.0


def test_talk_shares_a_personally_verified_resource_location(tmp_path):
    world, logger = _world(tmp_path)
    a0, a1 = world.agents
    a0.location = "east"
    world.resolve_action(a0, "gather")   # a0 now personally knows east has hunger
    a0.location = "hub"                  # back at hub, co-located with a1
    outcome = world.resolve_action(a0, "talk:a1")
    logger.close()

    assert a0.last_known_resource == ("hunger", "east")
    assert "east" in outcome
    assert any("east" in m and "a0 told you" in m for m in a1.recent_memories)
    # The listener now knows it too -- and could pass it on further.
    assert a1.last_known_resource == ("hunger", "east")


def test_informational_talk_never_leaks_self_model_text(tmp_path):
    # The whole point of a THIRD channel: 'self_model' remains the only mode
    # that ever copies identity text across agents.
    world, logger = _world(tmp_path)
    a0, a1 = world.agents
    a0.history.current.who_i_am = "a distinctive identity string"
    world.resolve_action(a0, "talk:a1")
    logger.close()
    assert not any("distinctive identity string" in m for m in a1.recent_memories)


def test_none_and_self_model_modes_are_unaffected(tmp_path):
    world, logger = _world(tmp_path, communication_mode="none")
    a0, a1 = world.agents
    a0.location = "east"
    world.resolve_action(a0, "gather")
    a0.location = "hub"
    outcome = world.resolve_action(a0, "talk:a1")
    logger.close()
    assert outcome == "made contact with a1"
    assert not any("east" in m for m in a1.recent_memories)
