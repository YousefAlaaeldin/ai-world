"""Root-run summaries for a candidate behavioral commitment (this build's
Phase 5): eligible-choice rate, survival/realization rate, and a comparison
of each Phase-4 branch against its ``sham`` reference at the same
checkpoint. Reuses aiworld.analysis.run_level rather than recomputing
per-run summaries from scratch.

Everything here is exploratory description, not evidence of intrinsic
purpose -- see docs/EXPERIMENT_PROTOCOL.md's evidence gates for what a
stronger "commitment" claim actually requires.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

from .behavioral_metrics import FEATURE_NAMES, behavioral_features_run
from .run_level import run_summary
from .trajectories import load_ticks

_OPPORTUNITY_FLAGS = (
    "can_consume", "can_gather", "can_build", "can_transfer",
    "can_take", "can_attack", "can_talk",
)


def eligible_choice_rate(run_dir: str) -> float:
    """Fraction of logged ticks where more than one kind of action beyond
    ``wait``/``move`` was actually available -- read directly from the
    already-logged ``TickRecord.pre_action_opportunities``, not new
    instrumentation. A run where every tick is a forced "nothing else to do
    but wait/move" is a run where "choice" has no content, regardless of
    what the agent happened to pick.
    """
    ticks = load_ticks(run_dir)
    if not ticks:
        return 0.0
    n_choice_points = sum(
        1 for record in ticks
        if any((record.get("pre_action_opportunities") or {}).get(flag)
              for flag in _OPPORTUNITY_FLAGS)
    )
    return n_choice_points / len(ticks)


def root_run_commitment_summary(run_dir: str) -> dict[str, Any]:
    """One run's commitment-relevant summary: eligible-choice rate plus the
    survival/realization/attrition fields already computed by
    ``run_level.run_summary``. Branch invariance (Phase 4) is a separate,
    per-checkpoint comparison -- see :func:`branch_invariance_summary`.
    """
    summary = run_summary(run_dir)
    return {
        "run_dir": run_dir,
        "eligible_choice_rate": eligible_choice_rate(run_dir),
        "survival_frac": summary["mean_survival_frac"],
        "valid_proposal_frac": summary["mean_valid_proposal_frac"],
        "parse_failure_frac": summary["mean_parse_failure_frac"],
        "successful_nonwait_frac": summary["mean_successful_nonwait_frac"],
        "behavioral_dispersion": summary["behavioral_dispersion"],
        "run_status": summary.get("run_status"),
    }


def branch_manifests(run_dir: str) -> list[dict[str, Any]]:
    """Load every branch manifest generated for one run
    (scripts.run_branches), if any. Returns [] for a run with no
    ``branches/`` directory."""
    pattern = os.path.join(run_dir, "branches", "*", "manifest.json")
    manifests = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            manifests.append(json.load(f))
    return manifests


def _focal_agent_features(branch_dir: str, focal_agent_id: str) -> dict[str, float] | None:
    if not os.path.exists(os.path.join(branch_dir, "ticks.jsonl")):
        return None
    try:
        features = behavioral_features_run(branch_dir)
    except ValueError:
        # A branch whose realized-behavior guard rejects it (e.g. no
        # completed status) contributes no comparison rather than crashing
        # the whole summary.
        return None
    return features.get(focal_agent_id)


def branch_invariance_summary(run_dir: str, focal_agent_id: str) -> dict[str, Any]:
    """Compare the focal agent's behavioral feature vector in every
    non-sham branch against the ``sham`` branch at the SAME checkpoint --
    "invariance under an inert perturbation" is what a ``sham`` branch is
    supposed to demonstrate; a real perturbation branch's distance from it
    is the effect size a probe measures.

    Exploratory only: this reports distances, not a claim. A stability/
    persistence CLAIM needs the evidence-gate discipline in
    docs/EXPERIMENT_PROTOCOL.md (run-level uncertainty, held-out
    replication), not just this per-run comparison.
    """
    manifests = branch_manifests(run_dir)
    by_checkpoint: dict[int, dict[str, dict[str, Any]]] = {}
    for manifest in manifests:
        by_checkpoint.setdefault(manifest["checkpoint_tick"], {})[
            manifest["branch_type"]] = manifest

    comparisons: list[dict[str, Any]] = []
    for checkpoint_tick, by_type in sorted(by_checkpoint.items()):
        sham_manifest = by_type.get("sham")
        if sham_manifest is None:
            continue
        sham_dir = os.path.join(run_dir, "branches", sham_manifest["branch_id"])
        sham_features = _focal_agent_features(sham_dir, focal_agent_id)
        if sham_features is None:
            continue
        sham_vector = np.array([sham_features[f] for f in FEATURE_NAMES])

        for branch_type, manifest in by_type.items():
            if branch_type == "sham":
                continue
            branch_dir = os.path.join(run_dir, "branches", manifest["branch_id"])
            branch_features = _focal_agent_features(branch_dir, focal_agent_id)
            if branch_features is None:
                continue
            branch_vector = np.array([branch_features[f] for f in FEATURE_NAMES])
            comparisons.append({
                "checkpoint_tick": checkpoint_tick,
                "branch_type": branch_type,
                "distance_from_sham": float(np.linalg.norm(branch_vector - sham_vector)),
            })

    n_checkpoints_with_sham = sum(
        1 for by_type in by_checkpoint.values() if "sham" in by_type)
    return {
        "focal_agent_id": focal_agent_id,
        "n_checkpoints_with_sham": n_checkpoints_with_sham,
        "comparisons": comparisons,
    }
