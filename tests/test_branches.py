"""Phase 4: branch perturbation generation, budget enforcement, and the
sham-branch determinism property at the branch-manifest level.
"""

from __future__ import annotations

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.experiments.branches import (  # noqa: E402
    BRANCH_TYPES,
    apply_intervention,
    branch_call_budget,
    eligible_now,
    enforce_call_budget,
    generate_branches,
    predeclare_checkpoints_eligible,
    predeclare_checkpoints_fixed,
    run_branch,
)
from aiworld.experiments.specs import ProbeSpec  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.seed_plan import SeedPlan  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402
from scripts.run_branches import run_branches_for_condition  # noqa: E402


def _agent(agent_id: str, location: str = "north") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def _tiny_cfg(ticks=6, n_agents=2):
    return {
        "world": {
            "locations": ["north", "south"],
            "edges": [["north", "south"]],
            "resources": [
                {"location": "north", "drive": "hunger", "quantity": 20},
                {"location": "south", "drive": "thirst", "quantity": 20},
            ],
            "mortality": True,
            "vocabulary": "neutral",
            "memory_style": "factual",
            "resource_map_mode": "fixed",
            "randomize_opaque_labels": False,
        },
        "agents": {"count": n_agents, "start_mode": "balanced"},
        "drives": {
            "hunger": {"decay_per_tick": 0.02, "replenish_amount": 0.6, "lethal": True},
            "thirst": {"decay_per_tick": 0.02, "replenish_amount": 0.6, "lethal": True},
            "affiliation": {"decay_per_tick": 0.01, "replenish_amount": 0.4, "lethal": False},
        },
        "reflection": {"every_n_ticks": 0, "enabled": False, "crisis_triggered": False,
                       "prompt": "flat"},
        "communication": {"mode": "none"},
        "run": {"ticks": ticks, "seeds": [0], "model": None},
    }


# ---- call budget ----

def test_branch_call_budget_multiplies_all_four_factors():
    assert branch_call_budget(3, 4, 5, 1) == 60
    assert branch_call_budget(0, 4, 5, 1) == 0


def test_branch_call_budget_rejects_negative_inputs():
    with pytest.raises(ValueError):
        branch_call_budget(-1, 1, 1, 1)


def test_enforce_call_budget_refuses_when_over_max_calls():
    with pytest.raises(ValueError, match="exceeds --max-calls"):
        enforce_call_budget(10, 4, 5, 1, max_calls=100)  # budget = 200


def test_enforce_call_budget_allows_when_within_max_calls():
    assert enforce_call_budget(2, 4, 5, 1, max_calls=100) == 40


def test_enforce_call_budget_allows_unlimited_when_max_calls_none():
    assert enforce_call_budget(100, 4, 5, 1, max_calls=None) == 2000


# ---- checkpoint predeclaration ----

def test_predeclare_checkpoints_fixed_schedule():
    assert predeclare_checkpoints_fixed(20, 5) == [5, 10, 15, 20]
    assert predeclare_checkpoints_fixed(4, 10) == []


def test_predeclare_checkpoints_fixed_rejects_bad_interval():
    with pytest.raises(ValueError):
        predeclare_checkpoints_fixed(10, 0)


def test_predeclare_checkpoints_eligible_from_logged_ticks():
    predicate = {"pre_action_opportunities.can_attack": ["truthy", True]}
    ticks = [
        {"tick": 1, "agent_id": "a0", "pre_action_opportunities": {"can_attack": False}},
        {"tick": 2, "agent_id": "a0", "pre_action_opportunities": {"can_attack": True}},
        {"tick": 2, "agent_id": "a1", "pre_action_opportunities": {"can_attack": True}},
        {"tick": 3, "agent_id": "a0", "pre_action_opportunities": {"can_attack": True}},
    ]
    assert predeclare_checkpoints_eligible(ticks, "a0", predicate) == [2, 3]


def test_eligible_now_reads_live_world_state(tmp_path):
    logger = RunLogger("elig1", root=str(tmp_path))
    agents = [_agent("a0", "north"), _agent("a1", "north")]
    world = World(["north"], [], agents, logger, mortality=False)
    predicate = {"pre_action_opportunities.can_talk": ["truthy", True]}
    result = eligible_now(predicate, world, world.agents[0])
    logger.close()
    assert result is True  # a1 is co-located, so talk is available


# ---- apply_intervention ----

def _build_snapshot(tmp_path, run_id="src"):
    logger = RunLogger(run_id, root=str(tmp_path))
    agents = [_agent("a0", "north"), _agent("a1", "south")]
    agents[0].recent_memories = ["tick 1: did something"]
    agents[0].long_term_memory = ["tick 1: a memorable event"]
    resources = [ResourceNode("north", DriveKind.HUNGER, quantity=5)]
    adjacency = {"north": {"south"}, "south": {"north"}}
    world = World(["north", "south"], resources, agents, logger,
                  rng=random.Random(1), mortality=True, adjacency=adjacency)
    snap = world.snapshot()
    logger.close()
    return snap


def test_apply_intervention_sham_makes_no_change(tmp_path):
    snap = _build_snapshot(tmp_path)
    branched, diff = apply_intervention(snap, "sham", "a0")
    assert branched.to_dict() == snap.to_dict()
    assert diff["branch_type"] == "sham"


def test_apply_intervention_means_obstacle_removes_one_edge(tmp_path):
    snap = _build_snapshot(tmp_path)
    branched, diff = apply_intervention(snap, "means_obstacle", "a0")
    assert diff["removed_edge"] == ["north", "south"]
    assert "south" not in branched.adjacency["north"]
    assert "north" not in branched.adjacency["south"]
    # Source snapshot must be untouched.
    assert "south" in snap.adjacency["north"]


def test_apply_intervention_cost_tradeoff_doubles_decay(tmp_path):
    snap = _build_snapshot(tmp_path)
    branched, diff = apply_intervention(snap, "cost_tradeoff", "a0")
    focal = next(a for a in branched.agents if a.agent_id == "a0")
    old_decay, new_decay = diff["decay_per_tick"]
    assert new_decay == old_decay * 2
    assert focal.drive_specs[diff["drive"]]["decay_per_tick"] == new_decay
    # Other agent and the source snapshot are untouched.
    other = next(a for a in branched.agents if a.agent_id == "a1")
    assert other.drive_specs == next(a for a in snap.agents if a.agent_id == "a1").drive_specs


def test_apply_intervention_ledger_ablation_empties_memory(tmp_path):
    snap = _build_snapshot(tmp_path)
    branched, diff = apply_intervention(snap, "ledger_ablation", "a0")
    focal = next(a for a in branched.agents if a.agent_id == "a0")
    assert focal.recent_memories == []
    assert focal.long_term_memory == []
    assert diff["removed_recent_memories"] == 1
    assert diff["removed_long_term_memory"] == 1
    # Source snapshot untouched.
    source_focal = next(a for a in snap.agents if a.agent_id == "a0")
    assert source_focal.recent_memories == ["tick 1: did something"]


def test_apply_intervention_rejects_deferred_branch_type(tmp_path):
    snap = _build_snapshot(tmp_path)
    with pytest.raises(ValueError, match="deferred"):
        apply_intervention(snap, "irrelevant_perturbation", "a0")


def test_apply_intervention_rejects_unknown_focal_agent(tmp_path):
    snap = _build_snapshot(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        apply_intervention(snap, "sham", "no_such_agent")


# ---- generate_branches ----

def test_generate_branches_produces_one_manifest_per_type_with_distinct_seeds(tmp_path):
    snap = _build_snapshot(tmp_path)
    seed_plan = SeedPlan(run_seed=5)
    results = generate_branches("parent1", snap, 10, "a0", seed_plan)

    assert len(results) == len(BRANCH_TYPES)
    branch_types = {m.branch_type for m, _ in results}
    assert branch_types == set(BRANCH_TYPES)
    seeds = [m.branch_seed for m, _ in results]
    assert len(set(seeds)) == len(seeds)  # every branch gets its own seed
    assert all(m.checkpoint_tick == 10 for m, _ in results)
    assert all(m.parent_run_id == "parent1" for m, _ in results)


def test_generate_branches_rejects_unknown_mode(tmp_path):
    snap = _build_snapshot(tmp_path)
    with pytest.raises(ValueError, match="branch mode"):
        generate_branches("parent1", snap, 10, "a0", SeedPlan(run_seed=0), mode="chaotic")


# ---- the sham-branch determinism property, at branch-manifest granularity ----

def test_sham_branch_reproduces_parent_next_tick_exactly(tmp_path):
    def policy(agent, perceived, world):
        return "wait"

    parent_logger = RunLogger("parent_branch", root=str(tmp_path))
    agents = [_agent("a0", "north"), _agent("a1", "south")]
    world = World(["north", "south"], [ResourceNode("north", DriveKind.HUNGER, quantity=5)],
                  agents, parent_logger, rng=random.Random(2), mortality=True,
                  adjacency={"north": {"south"}, "south": {"north"}})
    world.run(policy, ticks=2, reflect_every=0, reflect_fn=None)
    snap = world.snapshot()
    world.step(policy, reflect_every=0, reflect_fn=None)  # parent's real next tick
    parent_ticks_after = world.tick_count
    parent_logger.close()

    seed_plan = SeedPlan(run_seed=2)
    manifest, branched_snapshot = generate_branches(
        "parent_branch", snap, snap.tick_count + 1, "a0", seed_plan,
        branch_types=("sham",))[0]

    branch_logger = RunLogger("sham_branch", root=str(tmp_path))
    branch_world = run_branch(manifest, branched_snapshot, horizon=1, logger=branch_logger,
                              focal_policy=policy)
    branch_logger.close()

    assert branch_world.tick_count == parent_ticks_after
    for a, b in zip(sorted(world.agents, key=lambda a: a.agent_id),
                    sorted(branch_world.agents, key=lambda a: a.agent_id)):
        assert a.drives.snapshot() == b.drives.snapshot()
        assert a.location == b.location
        assert a.drives.alive == b.drives.alive


# ---- run_branches_for_condition (integration) ----

def test_run_branches_for_condition_fixed_checkpoints(tmp_path):
    cfg = _tiny_cfg(ticks=4, n_agents=2)
    probe = ProbeSpec(
        id="test_probe", construct="x", eligible_state_predicate={},
        intervention="", candidate_action="wait", required_viable_alternative="wait",
        scoring_rule="", branch_mode="microprobe", split="development",
        status="active", content_hash="x")

    summary = run_branches_for_condition(
        cfg, "int_fixed", probe=probe, focal_agent_id="agent_0", seed=0,
        checkpoint_mode="fixed", checkpoint_every_n=2, branch_horizon=1,
        branch_types=("sham", "ledger_ablation"), max_calls=1000,
        logs_root=str(tmp_path))

    # Checkpoints at tick 2 and 4 -> 2 checkpoints x 2 branch types = 4 branches.
    assert summary["n_branches"] == 4
    assert len(summary["branches"]) == 4
    checkpoint_ticks = sorted({b["checkpoint_tick"] for b in summary["branches"]})
    assert checkpoint_ticks == [2, 4]

    for branch in summary["branches"]:
        manifest_path = os.path.join(str(tmp_path), "int_fixed", "branches",
                                     branch["branch_id"], "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path, encoding="utf-8") as f:
            assert json.load(f)["branch_id"] == branch["branch_id"]

    meta_path = os.path.join(str(tmp_path), "int_fixed", "status.json")
    with open(meta_path, encoding="utf-8") as f:
        status_payload = json.load(f)
    assert status_payload["status"] == "completed"


def test_run_branches_for_condition_refuses_over_budget(tmp_path):
    cfg = _tiny_cfg(ticks=10, n_agents=2)
    probe = ProbeSpec(
        id="test_probe", construct="x", eligible_state_predicate={},
        intervention="", candidate_action="wait", required_viable_alternative="wait",
        scoring_rule="", branch_mode="microprobe", split="development",
        status="active", content_hash="x")

    with pytest.raises(ValueError, match="exceeds --max-calls"):
        run_branches_for_condition(
            cfg, "int_over_budget", probe=probe, focal_agent_id="agent_0", seed=0,
            checkpoint_mode="fixed", checkpoint_every_n=1, branch_horizon=5,
            max_calls=1, logs_root=str(tmp_path))

    # Refused before any branch/run directory was created.
    assert not os.path.exists(os.path.join(str(tmp_path), "int_over_budget"))


def test_run_branches_for_condition_eligible_mode_fires_on_live_predicate(tmp_path):
    cfg = _tiny_cfg(ticks=3, n_agents=2)
    probe = ProbeSpec(
        id="test_probe", construct="x",
        eligible_state_predicate={"pre_action_opportunities.can_consume": ["truthy", True]},
        intervention="", candidate_action="wait", required_viable_alternative="wait",
        scoring_rule="", branch_mode="microprobe", split="development",
        status="active", content_hash="x")

    summary = run_branches_for_condition(
        cfg, "int_eligible", probe=probe, focal_agent_id="agent_0", seed=0,
        checkpoint_mode="eligible", branch_horizon=1, branch_types=("sham",),
        max_calls=1000, logs_root=str(tmp_path))

    # agent_0 starts at a location with a hunger resource under balanced
    # placement (see _tiny_cfg), so can_consume is true from tick 1.
    assert summary["n_branches"] >= 1
