"""Phase 1: World.step() must be exactly what World.run() loops over -- not a
parallel, slightly-different implementation. run()'s own behavior (early stop
on no living agents, extinction logging) must be unchanged by the refactor.
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, TickResult, World  # noqa: E402


def _agent(agent_id: str, location: str = "site") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def _read_ticks(tmp_path, run_id):
    with open(tmp_path / run_id / "ticks.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _wander_policy(agent, perceived, world):
    neighbors = sorted(world.adjacency.get(agent.location, ()))
    return f"move:{neighbors[0]}" if neighbors else "wait"


def test_step_called_n_times_matches_run_ticks_n(tmp_path):
    """run(ticks=N) must be exactly N step() calls, byte-for-byte identical
    logged output -- the refactor's central guarantee."""
    resources = [ResourceNode("north", DriveKind.HUNGER, quantity=10)]

    run_logger = RunLogger("via_run", root=str(tmp_path))
    run_world = World(["north", "south"], resources,
                      [_agent("a0", "north"), _agent("a1", "south")],
                      run_logger, rng=random.Random(2), mortality=True,
                      adjacency={"north": {"south"}, "south": {"north"}})
    run_world.run(_wander_policy, ticks=4, reflect_every=0, reflect_fn=None)
    run_logger.close()

    step_logger = RunLogger("via_step", root=str(tmp_path))
    step_world = World(["north", "south"], [ResourceNode("north", DriveKind.HUNGER, quantity=10)],
                       [_agent("a0", "north"), _agent("a1", "south")],
                       step_logger, rng=random.Random(2), mortality=True,
                       adjacency={"north": {"south"}, "south": {"north"}})
    for _ in range(4):
        step_world.step(_wander_policy, reflect_every=0, reflect_fn=None)
    step_logger.close()

    def _without_run_id(records):
        return [{k: v for k, v in r.items() if k != "run_id"} for r in records]

    via_run = _without_run_id(_read_ticks(tmp_path, "via_run"))
    via_step = _without_run_id(_read_ticks(tmp_path, "via_step"))
    assert via_run == via_step
    assert run_world.tick_count == step_world.tick_count == 4


def test_step_returns_tick_result_with_expected_fields(tmp_path):
    logger = RunLogger("result_fields", root=str(tmp_path))
    world = World(["site"], [], [_agent("a0"), _agent("a1")], logger, mortality=False)

    result = world.step(lambda agent, perceived, world: "wait", reflect_every=0)
    logger.close()

    assert isinstance(result, TickResult)
    assert result.tick == 1
    assert sorted(result.live_at_start) == ["a0", "a1"]
    assert result.deaths == []
    assert result.crises_crossed == []
    assert result.extinct is False


def test_step_reports_extinction_and_deaths(tmp_path):
    logger = RunLogger("extinction", root=str(tmp_path))
    agent = _agent("a0")
    world = World(["site"], [], [agent], logger, mortality=True)
    # Force the agent to death threshold so the very first drive-decay step kills it.
    for state in agent.drives.states.values():
        state.level = 1.0

    result = world.step(lambda agent, perceived, world: "wait", reflect_every=0)
    logger.close()

    assert result.deaths == ["a0"]
    assert result.extinct is True

    with open(tmp_path / "extinction" / "events.jsonl", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert any(e["kind"] == "death" for e in events)
    assert any(e["kind"] == "extinction" for e in events)


def test_step_on_no_living_agents_still_increments_tick_and_returns_extinct(tmp_path):
    """Preserves the pre-refactor quirk exactly: tick_count increments even on
    a tick where every agent was already dead at the start."""
    logger = RunLogger("already_dead", root=str(tmp_path))
    agent = _agent("a0")
    world = World(["site"], [], [agent], logger, mortality=True)
    agent.drives.alive = False

    result = world.step(lambda agent, perceived, world: "wait", reflect_every=0)
    logger.close()

    assert world.tick_count == 1
    assert result.tick == 1
    assert result.live_at_start == []
    assert result.extinct is True


def test_run_stops_when_step_reports_extinction(tmp_path):
    logger = RunLogger("run_stops", root=str(tmp_path))
    agent = _agent("a0")
    world = World(["site"], [], [agent], logger, mortality=True)
    for state in agent.drives.states.values():
        state.level = 1.0

    world.run(lambda agent, perceived, world: "wait", ticks=10, reflect_every=0)
    logger.close()

    # Died on tick 1; run() must not have kept going for the other 9 ticks.
    assert world.tick_count == 1
