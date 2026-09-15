"""Phase 5: root-run commitment summaries and branch-vs-sham invariance."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.commitment_metrics import (  # noqa: E402
    branch_invariance_summary,
    branch_manifests,
    eligible_choice_rate,
    root_run_commitment_summary,
)
from aiworld.logging_schema import RunLogger, TickRecord  # noqa: E402
from aiworld.provenance import ACTION_API_VERSION, ANALYSIS_SCHEMA_VERSION  # noqa: E402
from aiworld.experiments.specs import ProbeSpec  # noqa: E402
from scripts.run_branches import run_branches_for_condition  # noqa: E402


def _log_run(root, run_id, records, *, finalize=True, meta=None):
    logger = RunLogger(run_id, root=str(root))
    if meta is not None:
        logger.log_meta(meta)
    for r in records:
        logger.log_tick(TickRecord(
            run_id=run_id, tick=r["tick"], agent_id=r["agent_id"], alive=True,
            drives=r.get("drives", {"hunger": 0.1, "thirst": 0.1}),
            position=r.get("position", "L1"), chosen_action=r["chosen_action"],
            perceived="", self_model_version=0,
            proposed_action=r.get("proposed_action", r["chosen_action"]),
            pre_action_drives=r.get("pre_action_drives", {"hunger": 0.1, "thirst": 0.1}),
            pre_action_position=r.get("pre_action_position", r.get("position", "L1")),
            pre_action_opportunities=r.get("pre_action_opportunities"),
            proposal_valid=True, action_succeeded=True,
            outcome_status="success", outcome="did something",
        ))
    if finalize:
        logger.finalize(status="completed", expected_ticks=len(records),
                        observed_ticks=len(records),
                        action_api_version=ACTION_API_VERSION,
                        analysis_schema_version=ANALYSIS_SCHEMA_VERSION)
    else:
        logger.close()
    return os.path.join(str(root), run_id)


def test_eligible_choice_rate_counts_ticks_with_a_real_option(tmp_path):
    records = [
        {"tick": 1, "agent_id": "a0", "chosen_action": "wait",
         "pre_action_opportunities": {"can_consume": True}},
        {"tick": 2, "agent_id": "a0", "chosen_action": "wait",
         "pre_action_opportunities": {}},  # no option beyond wait/move
    ]
    run_dir = _log_run(tmp_path, "elig", records)
    assert eligible_choice_rate(run_dir) == 0.5


def test_eligible_choice_rate_zero_on_empty_run(tmp_path):
    run_dir = _log_run(tmp_path, "empty", [])
    assert eligible_choice_rate(run_dir) == 0.0


def test_root_run_commitment_summary_has_expected_fields(tmp_path):
    records = [
        {"tick": 1, "agent_id": "a0", "chosen_action": "consume",
         "pre_action_opportunities": {"can_consume": True}},
        {"tick": 2, "agent_id": "a0", "chosen_action": "wait",
         "pre_action_opportunities": {}},
    ]
    run_dir = _log_run(tmp_path, "commit", records, meta={
        "run_id": "commit", "n_agents": 1,
        "resolved_config": {"run": {"ticks": 2}, "world": {"locations": ["L1"]}},
    })
    summary = root_run_commitment_summary(run_dir)
    assert summary["run_dir"] == run_dir
    assert summary["eligible_choice_rate"] == 0.5
    assert summary["run_status"] == "completed"
    assert 0.0 <= summary["survival_frac"] <= 1.0


def test_branch_manifests_returns_empty_for_a_run_with_no_branches(tmp_path):
    run_dir = _log_run(tmp_path, "no_branches", [
        {"tick": 1, "agent_id": "a0", "chosen_action": "wait"}])
    assert branch_manifests(run_dir) == []


def _tiny_cfg(ticks=4, n_agents=2):
    return {
        "world": {
            "locations": ["north", "south"],
            "edges": [["north", "south"]],
            "resources": [{"location": "north", "drive": "hunger", "quantity": 20}],
            "mortality": True, "vocabulary": "neutral", "memory_style": "factual",
            "resource_map_mode": "fixed", "randomize_opaque_labels": False,
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


def test_branch_invariance_summary_against_real_generated_branches(tmp_path):
    """Integration: run a real branch generation (Phase 4), then check Phase
    5's invariance summary actually finds the sham branch and compares every
    other branch type against it."""
    cfg = _tiny_cfg(ticks=2, n_agents=2)
    probe = ProbeSpec(
        id="test_probe", construct="x", eligible_state_predicate={},
        intervention="", candidate_action="wait", required_viable_alternative="wait",
        scoring_rule="", branch_mode="microprobe", split="development",
        status="active", content_hash="x")

    summary = run_branches_for_condition(
        cfg, "invariance_src", probe=probe, focal_agent_id="agent_0", seed=0,
        checkpoint_mode="fixed", checkpoint_every_n=2, branch_horizon=1,
        branch_types=("sham", "means_obstacle", "ledger_ablation"),
        max_calls=1000, logs_root=str(tmp_path))
    assert summary["n_branches"] == 3  # one checkpoint (tick 2) x 3 branch types

    run_dir = os.path.join(str(tmp_path), "invariance_src")
    manifests = branch_manifests(run_dir)
    assert len(manifests) == 3
    assert {m["branch_type"] for m in manifests} == {
        "sham", "means_obstacle", "ledger_ablation"}

    result = branch_invariance_summary(run_dir, "agent_0")
    assert result["focal_agent_id"] == "agent_0"
    assert result["n_checkpoints_with_sham"] == 1
    branch_types_compared = {c["branch_type"] for c in result["comparisons"]}
    assert branch_types_compared == {"means_obstacle", "ledger_ablation"}
    for comparison in result["comparisons"]:
        assert comparison["distance_from_sham"] >= 0.0


def test_branch_invariance_summary_empty_without_a_sham_branch(tmp_path):
    cfg = _tiny_cfg(ticks=2, n_agents=2)
    probe = ProbeSpec(
        id="test_probe", construct="x", eligible_state_predicate={},
        intervention="", candidate_action="wait", required_viable_alternative="wait",
        scoring_rule="", branch_mode="microprobe", split="development",
        status="active", content_hash="x")

    run_branches_for_condition(
        cfg, "no_sham_src", probe=probe, focal_agent_id="agent_0", seed=0,
        checkpoint_mode="fixed", checkpoint_every_n=2, branch_horizon=1,
        branch_types=("means_obstacle",), max_calls=1000, logs_root=str(tmp_path))

    run_dir = os.path.join(str(tmp_path), "no_sham_src")
    result = branch_invariance_summary(run_dir, "agent_0")
    assert result["n_checkpoints_with_sham"] == 0
    assert result["comparisons"] == []
