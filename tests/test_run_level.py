"""Run-level inference guards: a world is the replicate, not its agents."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.run_level import (  # noqa: E402
    paired_condition_comparison,
    run_summary,
    seed_run_dirs,
)
from aiworld.logging_schema import RunLogger, TickRecord  # noqa: E402


def _write_run(root, condition: str, seed: int, *, divergent: bool) -> str:
    run_id = f"{condition}/seed_{seed}"
    logger = RunLogger(run_id, root=str(root))
    logger.log_meta({
        "run_id": run_id,
        "policy": "stub",
        "n_agents": 2,
        "resolved_config": {
            "run": {"ticks": 2},
            "world": {"locations": ["L1"]},
        },
    })
    for tick in (1, 2):
        for agent_id, action in (
            ("a0", "consume" if divergent else "wait"),
            ("a1", "wait"),
        ):
            logger.log_tick(TickRecord(
                run_id=run_id,
                tick=tick,
                agent_id=agent_id,
                alive=True,
                drives={"hunger": 0.1, "thirst": 0.1},
                position="L1",
                chosen_action=action,
                perceived="",
                self_model_version=0,
                proposed_action=action,
                pre_action_drives={"hunger": 0.1, "thirst": 0.1},
                pre_action_position="L1",
                proposal_valid=True,
                action_succeeded=True,
                outcome_status="success" if action != "wait" else "no_effect",
                outcome="consumed R1" if action == "consume" else "waited",
            ))
    logger.close()
    return str(root / run_id)


def test_run_summary_keeps_agents_nested_inside_one_world(tmp_path):
    run_dir = _write_run(tmp_path, "left", 0, divergent=True)

    summary = run_summary(run_dir)

    assert summary["n_agents"] == 2
    assert summary["behavioral_dispersion"] > 0
    assert summary["mean_frac_consume"] == pytest.approx(0.5)


def test_paired_comparison_matches_seeds_and_reports_world_level_uncertainty(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_run(tmp_path, "left", 0, divergent=True)
    _write_run(tmp_path, "left", 1, divergent=True)
    _write_run(tmp_path, "left", 2, divergent=True)  # intentionally unmatched
    _write_run(tmp_path, "right", 0, divergent=False)
    _write_run(tmp_path, "right", 1, divergent=False)

    result = paired_condition_comparison(
        str(left_root), str(right_root),
        bootstrap_iterations=200, permutation_iterations=200, random_state=3,
    )

    assert result["n_matched_world_runs"] == 2
    assert result["matched_seeds"] == ["seed_0", "seed_1"]
    assert result["left_only_seeds"] == ["seed_2"]
    assert result["right_only_seeds"] == []
    assert result["mean_difference"] > 0
    assert len(result["bootstrap_95_ci"]) == 2
    assert 0 <= result["paired_sign_permutation_pvalue"] <= 1
    assert set(seed_run_dirs(str(left_root))) == {"seed_0", "seed_1", "seed_2"}


def test_paired_comparison_rejects_unknown_metric_and_insufficient_pairs(tmp_path):
    _write_run(tmp_path, "left", 0, divergent=True)
    _write_run(tmp_path, "right", 0, divergent=False)

    with pytest.raises(ValueError, match="Unknown run-level metric"):
        paired_condition_comparison(str(tmp_path / "left"), str(tmp_path / "right"),
                                    metric="invented")
    with pytest.raises(ValueError, match="at least two matched"):
        paired_condition_comparison(str(tmp_path / "left"), str(tmp_path / "right"))
