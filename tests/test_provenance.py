"""Phase 0: run integrity, completion status, and provenance-compatibility guards.

Two acceptance properties from docs/EXPERIMENT_PROTOCOL.md this locks in:
  - "A partial run cannot enter condition analysis" (require_completed_run).
  - "Two incompatible sweeps cannot be compared" (require_compatible_pair, wired
    into run_level.paired_condition_comparison).
Plus RunLogger.finalize()'s write-once/status-gating contract and SeedPlan's
determinism, which those guards and the sampling-null seed fix depend on.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.provenance import (  # noqa: E402
    load_run_provenance,
    require_compatible_pair,
    require_completed_run,
)
from aiworld.analysis.behavioral_metrics import behavioral_features_run  # noqa: E402
from aiworld.analysis.run_level import paired_condition_comparison  # noqa: E402
from aiworld.logging_schema import RunLogger, TickRecord  # noqa: E402
from aiworld.provenance import (  # noqa: E402
    ACTION_API_VERSION, ANALYSIS_SCHEMA_VERSION, git_revision)
from aiworld.seed_plan import SeedPlan, derive_seed  # noqa: E402


# ---- SeedPlan ----

def test_seed_plan_is_deterministic():
    plan = SeedPlan(run_seed=42)
    assert plan.seed_for("action", tick=1) == plan.seed_for("action", tick=1)


def test_seed_plan_distinguishes_every_keyword():
    plan = SeedPlan(run_seed=42)
    base = plan.seed_for("action")
    assert plan.seed_for("action", tick=1) != base
    assert plan.seed_for("action", tick=1) != plan.seed_for("action", tick=2)
    assert plan.seed_for("action", agent_id="a0") != plan.seed_for("action", agent_id="a1")
    assert plan.seed_for("action", branch_id="b0") != plan.seed_for("branch", branch_id="b0")
    assert plan.seed_for("action", slot=0) != plan.seed_for("action", slot=1)
    assert plan.seed_for("action", replicate=0) != plan.seed_for("action", replicate=1)


def test_seed_plan_differs_by_run_seed():
    assert SeedPlan(run_seed=1).seed_for("x") != SeedPlan(run_seed=2).seed_for("x")


def test_derive_seed_matches_seed_plan():
    assert derive_seed(7, "phase=x", "tick=None", "agent_id=None", "branch_id=None",
                       "slot=None", "replicate=None") == SeedPlan(run_seed=7).seed_for("x")


# ---- RunLogger.finalize ----

def _minimal_run(tmp_path, run_id="r1"):
    logger = RunLogger(run_id, root=str(tmp_path))
    logger.log_tick(TickRecord(
        run_id=run_id, tick=1, agent_id="a0", alive=True,
        drives={"hunger": 0.1}, position="L1", chosen_action="wait",
        perceived="", self_model_version=0,
    ))
    return logger


def test_finalize_writes_status_json_and_closes_streams(tmp_path):
    logger = _minimal_run(tmp_path)
    logger.finalize(status="completed", expected_ticks=1, observed_ticks=1,
                    action_api_version=ACTION_API_VERSION,
                    analysis_schema_version=ANALYSIS_SCHEMA_VERSION)

    path = os.path.join(str(tmp_path), "r1", "status.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["status"] == "completed"
    assert payload["expected_ticks"] == 1
    assert payload["observed_ticks"] == 1
    assert "ticks.jsonl" in payload["stream_sha256"]
    assert len(payload["stream_sha256"]["ticks.jsonl"]) == 64  # sha256 hex digest

    # close() after finalize() (which already closed the streams) must not error.
    logger.close()


def test_finalize_is_write_once(tmp_path):
    logger = _minimal_run(tmp_path)
    logger.finalize(status="completed")
    with pytest.raises(RuntimeError, match="immutable"):
        logger.finalize(status="completed")


def test_finalize_rejects_unknown_status(tmp_path):
    logger = _minimal_run(tmp_path)
    with pytest.raises(ValueError, match="Unknown run status"):
        logger.finalize(status="who_knows")


def test_finalize_records_git_revision_type(tmp_path):
    # Never raises; either a string or None depending on environment.
    revision = git_revision()
    assert revision is None or isinstance(revision, str)


# ---- require_completed_run: leniency on missing data, strictness on stated data ----

def test_require_completed_run_is_lenient_when_status_json_is_absent(tmp_path):
    logger = _minimal_run(tmp_path)
    logger.close()  # no finalize() -- matches every pre-Phase-0 test fixture

    provenance = require_completed_run(str(tmp_path / "r1"))
    assert provenance.get("status") is None


def test_require_completed_run_rejects_explicit_non_completed_status(tmp_path):
    logger = _minimal_run(tmp_path)
    logger.finalize(status="aborted", error="boom")

    with pytest.raises(ValueError, match="status 'aborted'"):
        require_completed_run(str(tmp_path / "r1"))


def test_require_completed_run_rejects_action_api_version_mismatch(tmp_path):
    logger = _minimal_run(tmp_path)
    logger.finalize(status="completed", action_api_version="v0")

    with pytest.raises(ValueError, match="action_api_version"):
        require_completed_run(str(tmp_path / "r1"),
                              expected_action_api_version="v1")


def test_require_completed_run_rejects_analysis_schema_version_mismatch(tmp_path):
    logger = _minimal_run(tmp_path)
    logger.finalize(status="completed", analysis_schema_version=0)

    with pytest.raises(ValueError, match="analysis_schema_version"):
        require_completed_run(str(tmp_path / "r1"),
                              expected_analysis_schema_version=1)


def test_require_completed_run_merges_meta_and_status(tmp_path):
    logger = _minimal_run(tmp_path)
    logger.log_meta({"run_id": "r1", "n_agents": 3})
    logger.finalize(status="completed", expected_ticks=5)

    provenance = load_run_provenance(str(tmp_path / "r1"))
    assert provenance["n_agents"] == 3
    assert provenance["status"] == "completed"
    assert provenance["expected_ticks"] == 5


def test_behavioral_features_run_rejects_an_aborted_run(tmp_path):
    """The acceptance property: a partial run cannot enter condition analysis."""
    logger = _minimal_run(tmp_path)
    logger.log_meta({"run_id": "r1", "n_agents": 1,
                     "resolved_config": {"run": {"ticks": 5},
                                        "world": {"locations": ["L1"]}}})
    logger.finalize(status="aborted", error="the process died mid-run")

    with pytest.raises(ValueError, match="status 'aborted'"):
        behavioral_features_run(str(tmp_path / "r1"))


# ---- require_compatible_pair ----

def _run_with_llm_meta(tmp_path, run_id, *, model, resource_map_mode="fixed"):
    logger = RunLogger(run_id, root=str(tmp_path))
    logger.log_meta({
        "run_id": run_id,
        "llm": {"model": model, "provider": "openai", "temperature": 0.8},
        "resource_map_mode": resource_map_mode,
        "initial_location_mode": "balanced",
        "action_api_version": ACTION_API_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
    })
    logger.finalize(status="completed")
    return str(tmp_path / run_id)


def test_require_compatible_pair_allows_matching_provenance(tmp_path):
    left = _run_with_llm_meta(tmp_path, "left", model="model-a")
    right = _run_with_llm_meta(tmp_path, "right", model="model-a")
    require_compatible_pair(left, right)  # must not raise


def test_require_compatible_pair_rejects_model_mismatch(tmp_path):
    left = _run_with_llm_meta(tmp_path, "left", model="model-a")
    right = _run_with_llm_meta(tmp_path, "right", model="model-b")
    with pytest.raises(ValueError, match="llm.model"):
        require_compatible_pair(left, right)


def test_require_compatible_pair_rejects_map_mode_mismatch(tmp_path):
    left = _run_with_llm_meta(tmp_path, "left", model="model-a",
                              resource_map_mode="fixed")
    right = _run_with_llm_meta(tmp_path, "right", model="model-a",
                               resource_map_mode="counterbalanced")
    with pytest.raises(ValueError, match="resource_map_mode"):
        require_compatible_pair(left, right)


def test_require_compatible_pair_skips_fields_missing_on_either_side(tmp_path):
    logger_left = RunLogger("left", root=str(tmp_path))
    logger_left.log_meta({"run_id": "left"})  # no "llm" key at all
    logger_left.finalize(status="completed")
    right = _run_with_llm_meta(tmp_path, "right", model="model-a")

    require_compatible_pair(str(tmp_path / "left"), right)  # must not raise


# ---- wired into paired_condition_comparison: "two incompatible sweeps cannot
# be compared" ----

def _write_sweep_run(root, condition: str, seed: int, *, model: str,
                     divergent: bool) -> None:
    run_id = f"{condition}/seed_{seed}"
    logger = RunLogger(run_id, root=str(root))
    logger.log_meta({
        "run_id": run_id,
        "llm": {"model": model, "provider": "openai", "temperature": 0.8},
        "resolved_config": {"run": {"ticks": 2}, "world": {"locations": ["L1"]}},
    })
    for tick in (1, 2):
        for agent_id, action in (
            ("a0", "consume" if divergent else "wait"),
            ("a1", "wait"),
        ):
            logger.log_tick(TickRecord(
                run_id=run_id, tick=tick, agent_id=agent_id, alive=True,
                drives={"hunger": 0.1, "thirst": 0.1}, position="L1",
                chosen_action=action, perceived="", self_model_version=0,
                proposed_action=action, pre_action_drives={"hunger": 0.1, "thirst": 0.1},
                pre_action_position="L1", proposal_valid=True, action_succeeded=True,
                outcome_status="success" if action != "wait" else "no_effect",
                outcome="consumed R1" if action == "consume" else "waited",
            ))
    logger.finalize(status="completed")


def test_paired_condition_comparison_rejects_a_different_model_pair(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_sweep_run(tmp_path, "left", 0, model="model-a", divergent=True)
    _write_sweep_run(tmp_path, "left", 1, model="model-a", divergent=True)
    _write_sweep_run(tmp_path, "right", 0, model="model-b", divergent=False)
    _write_sweep_run(tmp_path, "right", 1, model="model-b", divergent=False)

    with pytest.raises(ValueError, match="not a compatible pair"):
        paired_condition_comparison(str(left_root), str(right_root))


def test_paired_condition_comparison_allows_a_matching_model_pair(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    _write_sweep_run(tmp_path, "left", 0, model="model-a", divergent=True)
    _write_sweep_run(tmp_path, "left", 1, model="model-a", divergent=True)
    _write_sweep_run(tmp_path, "right", 0, model="model-a", divergent=False)
    _write_sweep_run(tmp_path, "right", 1, model="model-a", divergent=False)

    result = paired_condition_comparison(
        str(left_root), str(right_root),
        bootstrap_iterations=50, permutation_iterations=50)
    assert result["n_matched_world_runs"] == 2
