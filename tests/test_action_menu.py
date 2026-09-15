"""Phase 3: legal_action_menu completeness against World's own ground truth."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.actions import assign_opaque_ids, legal_action_menu, render_menu_prompt  # noqa: E402
from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402


def _agent(agent_id: str, location: str = "north") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def test_menu_always_includes_wait(tmp_path):
    logger = RunLogger("menu1", root=str(tmp_path))
    world = World(["north"], [], [_agent("a0")], logger, mortality=False)
    menu = legal_action_menu(world, world.agents[0])
    logger.close()
    assert "wait" in menu


def test_menu_matches_pre_action_opportunities_ground_truth(tmp_path):
    """No mechanically legal action is silently filtered from the menu, and
    no unavailable one is silently included -- checked directly against
    World._pre_action_opportunities(), the same computation TickRecord logs."""
    logger = RunLogger("menu2", root=str(tmp_path))
    agents = [_agent("a0", "north"), _agent("a1", "north")]
    agents[0].inventory[DriveKind.HUNGER] = 2
    resources = [ResourceNode("north", DriveKind.HUNGER, quantity=5)]
    adjacency = {"north": {"south"}, "south": {"north"}}
    world = World(["north", "south"], resources, agents, logger,
                  mortality=True, adjacency=adjacency)
    agent = world.agents[0]
    opportunities = world._pre_action_opportunities(agent)

    menu = legal_action_menu(world, agent)
    logger.close()

    assert ("consume" in menu) == opportunities["can_consume"]
    assert ("gather" in menu) == opportunities["can_gather"]
    assert ("build" in menu) == opportunities["can_build"]
    for loc in opportunities["reachable_locations"]:
        assert f"move:{world.location_label(loc)}" in menu
    for target in opportunities["transfer_targets"]:
        assert f"transfer:{target}" in menu
    for target in opportunities["take_targets"]:
        assert f"take:{target}" in menu
    for target in opportunities["attack_targets"]:
        assert f"attack:{target}" in menu
    for target in opportunities["talk_targets"]:
        assert f"talk:{target}" in menu
    # No prosocial/harmful action is filtered relative to another: build,
    # transfer, take, and attack are all present or absent purely from
    # mechanical availability, never omitted as a category.
    move_count = sum(1 for item in menu if item.startswith("move:"))
    assert move_count == len(opportunities["reachable_locations"])


def test_menu_excludes_attack_and_talk_when_alone(tmp_path):
    logger = RunLogger("menu3", root=str(tmp_path))
    world = World(["north"], [], [_agent("solo")], logger, mortality=False)
    menu = legal_action_menu(world, world.agents[0])
    logger.close()
    assert not any(item.startswith("attack:") for item in menu)
    assert not any(item.startswith("talk:") for item in menu)
    assert not any(item.startswith("take:") for item in menu)
    assert not any(item.startswith("transfer:") for item in menu)


def test_assign_opaque_ids_covers_every_item_uniquely(tmp_path):
    logger = RunLogger("menu4", root=str(tmp_path))
    agents = [_agent("a0", "north"), _agent("a1", "north")]
    world = World(["north"], [], agents, logger, mortality=False)
    menu = legal_action_menu(world, world.agents[0])
    logger.close()

    id_map = assign_opaque_ids(menu, random.Random(0))
    assert len(id_map) == len(menu)
    assert set(id_map.values()) == set(menu)
    assert len(id_map) == len(set(id_map))  # ids themselves are unique


def test_assign_opaque_ids_differs_across_decisions(tmp_path):
    logger = RunLogger("menu5", root=str(tmp_path))
    world = World(["north"], [], [_agent("a0")], logger, mortality=False)
    menu = legal_action_menu(world, world.agents[0])
    logger.close()

    rng = random.Random(0)
    first = assign_opaque_ids(menu, rng)
    second = assign_opaque_ids(menu, rng)
    assert first != second  # same canonical menu, different id assignment


def test_render_menu_prompt_lists_every_id_and_action():
    menu = ["wait", "move:south"]
    id_map = {"K7": "wait", "Q4": "move:south"}
    prompt = render_menu_prompt(menu, id_map)
    assert "K7: wait" in prompt
    assert "Q4: move:south" in prompt
    assert '{"choice_id":"<id>"}' in prompt
