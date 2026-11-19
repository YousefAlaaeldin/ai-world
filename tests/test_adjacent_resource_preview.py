"""Opt-in adjacent-resource preview (default off -- unchanged behavior).

Built after the Phase 0.5 feasibility pilot (2026-09-15): every traced dying
agent had a one-hop route to its crisis resource clearly listed by name, but
no visibility into what was actually THERE until arriving. This flag lets a
follow-up pilot separate "didn't know" from "knew and didn't act" -- without
it, World.observations_for() only ever names a neighbor, never its content.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402


def _agent(agent_id: str, location: str) -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def _build_world(tmp_path, adjacent_resource_preview: bool) -> World:
    logger = RunLogger("preview_test", root=str(tmp_path))
    agents = [_agent("a0", "north")]
    resources = [
        ResourceNode("north", DriveKind.HUNGER, quantity=3),
        ResourceNode("south", DriveKind.THIRST, quantity=7),
    ]
    adjacency = {"north": {"south"}, "south": {"north"}}
    return World(["north", "south"], resources, agents, logger,
                rng=random.Random(0), mortality=True, adjacency=adjacency,
                adjacent_resource_preview=adjacent_resource_preview)


def test_preview_disabled_by_default_shows_neighbor_name_only(tmp_path):
    world = _build_world(tmp_path, adjacent_resource_preview=False)
    texts = [obs.text for obs in world.observations_for(world.agents[0])]
    assert "you could move to south" in texts
    assert not any("quantity 7" in t for t in texts)


def test_preview_enabled_reveals_neighbor_resource_content(tmp_path):
    world = _build_world(tmp_path, adjacent_resource_preview=True)
    texts = [obs.text for obs in world.observations_for(world.agents[0])]
    assert "you could move to south" in texts
    # South's resource (thirst, neutral vocabulary -> "a maintenance source")
    # is now visible before the agent ever moves there.
    assert "at south there is a maintenance source (quantity 7)" in texts
    # Own location's resource (hunger -> "an energy source") is unaffected.
    assert "here there is an energy source (quantity 3)" in texts


def test_preview_flag_survives_snapshot_round_trip(tmp_path):
    world = _build_world(tmp_path, adjacent_resource_preview=True)
    snap = world.snapshot()
    assert snap.adjacent_resource_preview is True

    logger2 = RunLogger("preview_test_restored", root=str(tmp_path))
    restored = World.from_snapshot(snap, logger2, rng=random.Random(1))
    assert restored.adjacent_resource_preview is True
    texts = [obs.text for obs in restored.observations_for(restored.agents[0])]
    assert "at south there is a maintenance source (quantity 7)" in texts
