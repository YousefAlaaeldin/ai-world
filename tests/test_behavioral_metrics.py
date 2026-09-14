"""Tests for behavioral metrics (Tranche 3). Pure Python + numpy/sklearn; no LLM.

Verifies the fixed-schema feature extraction, the behavioral-divergence metric, and
the silhouette-based clustering with its null baseline -- the pieces that make the
de-primed baseline analyzable by BEHAVIOR rather than self-model text.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.logging_schema import RunLogger, TickRecord  # noqa: E402
from aiworld.analysis.behavioral_metrics import (  # noqa: E402
    agent_features, behavioral_features_run, behavioral_divergence_over_time,
    cluster_behaviors, feature_matrix, FEATURE_NAMES)

LOW = {"hunger": 0.1, "thirst": 0.1}
HIGH = {"hunger": 0.9, "thirst": 0.1}


def _rec(tick, agent_id, action, position, drives, alive=True, **extra):
    record = {"tick": tick, "agent_id": agent_id, "action": action,
              "position": position, "chosen_action": action, "drives": drives,
              "alive": alive}
    record.update(extra)
    return record


# ---- feature extraction ----

def test_always_wait_agent():
    recs = [_rec(i, "a0", "wait", "L1", LOW) for i in range(1, 6)]
    f = agent_features(recs, total_ticks=5, n_agents=1, all_locations={"L1"})
    assert f["frac_wait"] == 1.0
    assert f["frac_consume"] == 0.0 and f["frac_move"] == 0.0
    assert f["move_rate"] == 0.0
    assert f["discretionary_activity"] == 0.0   # had slack, chose to do nothing
    assert f["survival_frac"] == 1.0


def test_discretionary_activity_when_not_under_pressure():
    # Low drives every tick (slack), but the agent keeps moving -> high discretionary.
    recs = [_rec(i, "a0", f"move:L{i%2}", f"L{i%2}", LOW) for i in range(1, 6)]
    f = agent_features(recs, total_ticks=5, n_agents=1, all_locations={"L0", "L1"})
    assert f["discretionary_activity"] == 1.0
    assert f["move_rate"] == 1.0                 # position changed every step


def test_over_acquisition_only_counts_consume_without_need():
    needy = agent_features([_rec(1, "a0", "consume", "L1", HIGH)],
                           total_ticks=1, n_agents=1, all_locations={"L1"})
    greedy = agent_features([_rec(1, "a0", "consume", "L1", LOW)],
                            total_ticks=1, n_agents=1, all_locations={"L1"})
    assert needy["over_acquisition"] == 0.0      # consumed under real need
    assert greedy["over_acquisition"] == 1.0     # consumed with nothing needed


def test_social_partners_fraction():
    recs = [_rec(1, "a0", "talk:a1", "L1", LOW), _rec(2, "a0", "talk:a2", "L1", LOW)]
    f = agent_features(recs, total_ticks=2, n_agents=3, all_locations={"L1"})
    assert f["social_partners_frac"] == 1.0      # talked to both of the other 2
    assert f["successful_talk_rate"] == 1.0
    assert "talk_rate" not in FEATURE_NAMES       # was a duplicate of frac_talk


def test_metrics_use_pre_action_state_and_keep_parse_failures_out_of_action_mix():
    # Record 1 consumed under high PRE-action pressure, even though its post-action
    # state is low. Record 2 is a malformed LLM completion that fell back to wait.
    # Record 3 is a legal but failed consume attempt while slack.
    recs = [
        _rec(1, "a0", "consume", "L1", LOW,
             pre_action_drives=HIGH, proposed_action="consume", parse_status="ok",
             proposal_valid=True, outcome_status="success", action_succeeded=True),
        _rec(2, "a0", "wait", "L1", LOW,
             pre_action_drives=LOW, proposed_action="wait",
             parse_status="no_legal_action", proposal_valid=False,
             outcome_status="no_effect", action_succeeded=True),
        _rec(3, "a0", "consume", "L1", LOW,
             pre_action_drives=LOW, proposed_action="consume", parse_status="ok",
             proposal_valid=True, outcome_status="failed", action_succeeded=False),
    ]
    f = agent_features(recs, total_ticks=3, n_agents=1, all_locations={"L1"})
    assert f["frac_consume"] == 1.0
    assert f["frac_wait"] == 0.0                 # parser fallback is not a choice
    assert f["valid_proposal_frac"] == 2 / 3
    assert f["parse_failure_frac"] == 1 / 3
    assert f["successful_nonwait_frac"] == 1 / 3
    assert f["over_acquisition"] == 0.0          # uses pre-action, not post-action drive
    assert f["discretionary_activity"] == 0.0    # failed action is not realized activity


def test_successful_low_pressure_consume_is_over_acquisition():
    rec = _rec(1, "a0", "consume", "L1", HIGH,
               pre_action_drives=LOW, proposed_action="consume", parse_status="ok",
               proposal_valid=True, outcome_status="success", action_succeeded=True)
    f = agent_features([rec], total_ticks=1, n_agents=1, all_locations={"L1"})
    assert f["over_acquisition"] == 1.0


def test_features_have_fixed_schema():
    f = agent_features([_rec(1, "a0", "wait", "L1", LOW)], 1, 1, {"L1"})
    assert set(f.keys()) == set(FEATURE_NAMES)


# ---- run-level + divergence ----

def _log(run_dir_root, run_id, recs, meta=None):
    logger = RunLogger(run_id, root=run_dir_root)
    if meta is not None:
        logger.log_meta(meta)
    for r in recs:
        logger.log_tick(TickRecord(
            run_id=run_id, tick=r["tick"], agent_id=r["agent_id"], alive=r["alive"],
            drives=r["drives"], position=r["position"], chosen_action=r["chosen_action"],
            perceived="", self_model_version=0, cause_of_death=None,
            proposed_action=r.get("proposed_action"),
            pre_action_drives=r.get("pre_action_drives"),
            pre_action_position=r.get("pre_action_position"),
            pre_action_inventory=r.get("pre_action_inventory"),
            pre_action_opportunities=r.get("pre_action_opportunities"),
            parse_status=r.get("parse_status"), proposal_valid=r.get("proposal_valid"),
            outcome_status=r.get("outcome_status"),
            action_succeeded=r.get("action_succeeded"), outcome=r.get("outcome")))
    logger.close()
    return os.path.join(run_dir_root, run_id)


def test_behavioral_features_run(tmp_path):
    recs = ([_rec(i, "a0", "consume", "L1", HIGH) for i in range(1, 6)] +
            [_rec(i, "a1", "wait", "L2", LOW) for i in range(1, 6)])
    run_dir = _log(str(tmp_path), "r", recs)
    feats = behavioral_features_run(run_dir)
    assert feats["a0"]["frac_consume"] == 1.0
    assert feats["a1"]["frac_wait"] == 1.0


def test_features_use_configured_horizon_full_map_and_pre_move_position(tmp_path):
    recs = [_rec(1, "a0", "move:L2", "L2", LOW,
                 pre_action_position="L1", proposed_action="move:L2",
                 proposal_valid=True, action_succeeded=True, outcome_status="success")]
    run_dir = _log(str(tmp_path), "r", recs, meta={
        "run_id": "r",
        "n_agents": 1,
        "resolved_config": {
            "run": {"ticks": 10},
            "world": {"locations": ["L1", "L2", "L3"]},
        },
    })
    f = behavioral_features_run(run_dir)["a0"]
    assert f["survival_frac"] == 0.1             # planned horizon, not last observed tick
    assert f["exploration_coverage"] == 2 / 3    # full map and source+destination
    assert f["move_rate"] == 1.0                 # first-tick realized move counts


def test_behavioral_divergence_positive_when_agents_differ(tmp_path):
    recs = ([_rec(i, "a0", "consume", "L1", HIGH) for i in range(1, 11)] +
            [_rec(i, "a1", "wait", "L2", LOW) for i in range(1, 11)])
    run_dir = _log(str(tmp_path), "r", recs)
    div = behavioral_divergence_over_time(run_dir, window=5)
    assert div and all(v > 0 for v in div.values())   # they behave differently


# ---- clustering with null baseline ----

def test_cluster_separates_two_behavior_types(tmp_path):
    # 4 pure consumers + 4 pure movers -> a clean 2-cluster structure that must
    # beat the column-shuffled null and be picked over a forced larger k.
    recs = []
    for a in range(4):
        recs += [_rec(i, f"c{a}", "consume", "L1", HIGH) for i in range(1, 6)]
    for a in range(4):
        recs += [_rec(i, f"m{a}", f"move:L{i%3}", f"L{i%3}", LOW) for i in range(1, 6)]
    run_dir = _log(str(tmp_path), "r", recs)
    result = cluster_behaviors([run_dir])
    assert "error" not in result
    assert result["selected_k"] == 2
    assert result["clusters_beat_null"] is True
    # the two types must not share a cluster
    label_of = {}
    for cid, members in result["clusters"].items():
        for m in members:
            label_of[m["agent_id"]] = cid
    consumer_labels = {label_of[f"c{a}"] for a in range(4)}
    mover_labels = {label_of[f"m{a}"] for a in range(4)}
    assert consumer_labels.isdisjoint(mover_labels)
