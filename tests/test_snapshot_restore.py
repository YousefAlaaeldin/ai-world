"""Phase 1: World.snapshot() / World.from_snapshot() round-trip and determinism.

The property Phase 4's branch runner depends on: restoring a snapshot and
stepping it once with the same policy reproduces the parent world's next tick
exactly -- a "sham" branch (one with no actual perturbation) must be provably
inert, or a real perturbation branch's effect can't be trusted either.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModel, SelfModelHistory  # noqa: E402
from aiworld.sim.snapshot import WorldSnapshot  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402


def _agent(agent_id: str, location: str = "north") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def _build_world(tmp_path, run_id="parent", seed=5) -> World:
    logger = RunLogger(run_id, root=str(tmp_path))
    agents = [_agent("a0", "north"), _agent("a1", "south")]
    resources = [
        ResourceNode("north", DriveKind.HUNGER, quantity=5),
        ResourceNode("south", DriveKind.THIRST, quantity=5),
    ]
    adjacency = {"north": {"south"}, "south": {"north"}}
    return World(["north", "south"], resources, agents, logger,
                rng=random.Random(seed), mortality=True, adjacency=adjacency,
                memory_style="factual")


def _wait_policy(agent, perceived, world) -> str:
    return "wait"


def _mixed_policy(agent, perceived, world) -> str:
    """Deterministic given (agent_id, position) -- no hidden internal RNG, so
    a snapshot's saved world.rng state is the only source of randomness left,
    letting a restored branch reproduce the parent exactly."""
    if agent.agent_id == "a0":
        return "gather" if world.resources and world.resources[0].quantity > 0 else "wait"
    return f"move:{next(iter(world.adjacency.get(agent.location, ())), agent.location)}"


def test_snapshot_round_trip_preserves_state(tmp_path):
    world = _build_world(tmp_path)
    world.run(_mixed_policy, ticks=3, reflect_every=0, reflect_fn=None)

    snap = world.snapshot()
    restored_logger = RunLogger("restored_copy", root=str(tmp_path))
    restored = World.from_snapshot(snap, restored_logger)

    assert restored.tick_count == world.tick_count
    for a, b in zip(sorted(world.agents, key=lambda a: a.agent_id),
                    sorted(restored.agents, key=lambda a: a.agent_id)):
        assert a.agent_id == b.agent_id
        assert a.location == b.location
        assert a.drives.snapshot() == b.drives.snapshot()
        assert a.drives.alive == b.drives.alive
        assert dict(a.inventory) == {k: v for k, v in b.inventory.items()}
        assert list(a.recent_memories) == list(b.recent_memories)
        assert list(a.long_term_memory) == list(b.long_term_memory)
        assert [v.to_dict() for v in a.history.versions] == \
            [v.to_dict() for v in b.history.versions]
    assert [(r.location, r.drive, r.quantity) for r in world.resources] == \
        [(r.location, r.drive, r.quantity) for r in restored.resources]

    def _normalize_stock(stock):
        return {loc: {(k.value if isinstance(k, DriveKind) else str(k)): v
                      for k, v in per_loc.items()}
               for loc, per_loc in stock.items()}

    assert _normalize_stock(world.public_stock) == _normalize_stock(restored.public_stock)

    restored_logger.close()


def test_snapshot_payload_is_json_safe_and_round_trips_through_json(tmp_path):
    world = _build_world(tmp_path)
    world.run(_mixed_policy, ticks=2, reflect_every=0, reflect_fn=None)
    snap = world.snapshot()

    as_json = json.dumps(snap.to_dict())  # must not raise
    reloaded = WorldSnapshot.from_dict(json.loads(as_json))

    assert reloaded.tick_count == snap.tick_count
    assert reloaded.agents[0].agent_id == snap.agents[0].agent_id
    assert reloaded.rng_state == snap.rng_state

    # No file handles, logger, or client objects anywhere in the payload.
    def _all_leaves(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from _all_leaves(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _all_leaves(v)
        else:
            yield obj

    for leaf in _all_leaves(snap.to_dict()):
        assert isinstance(leaf, (str, int, float, bool, type(None)))


def test_restored_branch_reproduces_next_tick_exactly(tmp_path):
    """The sham-branch determinism property: snapshot -> restore -> step once
    with the SAME policy must reproduce the parent's own next step exactly,
    including its logged tick records."""
    parent = _build_world(tmp_path, run_id="parent2", seed=11)
    parent.run(_mixed_policy, ticks=3, reflect_every=0, reflect_fn=None)
    snap = parent.snapshot()

    # Continue the parent world one more tick.
    parent.step(_mixed_policy, reflect_every=0, reflect_fn=None)
    parent.logger.close()

    # Independently: restore from the snapshot and step once, same policy.
    restored_logger = RunLogger("restored_branch", root=str(tmp_path))
    restored = World.from_snapshot(snap, restored_logger)
    restored.step(_mixed_policy, reflect_every=0, reflect_fn=None)
    restored_logger.close()

    def _read_ticks(run_id):
        path = tmp_path / run_id / "ticks.jsonl"
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    parent_ticks = _read_ticks("parent2")
    restored_ticks = _read_ticks("restored_branch")

    # The restored branch only logged the ONE post-snapshot tick; compare it
    # against the parent's tick of the same number.
    last_parent_tick_num = restored_ticks[0]["tick"]
    last_parent_records = sorted(
        (r for r in parent_ticks if r["tick"] == last_parent_tick_num),
        key=lambda r: r["agent_id"])
    restored_records = sorted(restored_ticks, key=lambda r: r["agent_id"])

    assert len(last_parent_records) == len(restored_records) > 0
    compare_fields = ("agent_id", "alive", "drives", "position", "chosen_action",
                      "outcome", "outcome_status", "action_succeeded",
                      "proposal_valid", "proposal_index", "resolution_index")
    for parent_rec, restored_rec in zip(last_parent_records, restored_records):
        for field in compare_fields:
            assert parent_rec[field] == restored_rec[field], (
                f"field {field!r} diverged: {parent_rec[field]!r} != "
                f"{restored_rec[field]!r}")


def test_from_snapshot_accepts_an_independent_rng_for_a_real_branch(tmp_path):
    """A genuinely perturbed branch (Phase 4) supplies its own seeded rng
    instead of continuing the parent's stream -- from_snapshot must honor
    that override rather than always restoring the saved state."""
    parent = _build_world(tmp_path, run_id="parent3", seed=3)
    parent.run(_mixed_policy, ticks=2, reflect_every=0, reflect_fn=None)
    snap = parent.snapshot()
    parent.logger.close()

    branch_logger = RunLogger("independent_branch", root=str(tmp_path))
    branch_rng = random.Random(999)
    branch = World.from_snapshot(snap, branch_logger, rng=branch_rng)

    assert branch.rng is branch_rng
    branch_logger.close()
