"""Run-level summaries and paired condition comparisons.

Agents in one world share the same map, resources, action-resolution history, and
social environment.  They are therefore *nested observations*, not independent
experimental replicates.  This module makes the world run the unit used for
condition-level estimates and uncertainty.

It deliberately reports a small set of pre-declared, descriptive outcomes.  It
does not turn a cluster or a low p-value into evidence of intrinsic purpose; use the
claim/evidence gates in ``docs/EXPERIMENT_PROTOCOL.md`` when interpreting output.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable

import numpy as np

from .behavioral_metrics import FEATURE_NAMES, behavioral_features_run, feature_matrix
from .provenance import load_run_provenance, require_compatible_pair


RUN_LEVEL_METRICS = ("behavioral_dispersion", *(
    f"mean_{feature}" for feature in FEATURE_NAMES))


def _mean_pairwise_distance(matrix: np.ndarray) -> float:
    """Mean raw-feature distance between agents within one world.

    All current behavioral features are bounded rates/fractions, so raw distance
    preserves their declared scale.  We intentionally do not z-score inside each
    run: doing so would manufacture relative divergence even when the absolute
    behavioral variation is negligible.
    """
    n = len(matrix)
    if n < 2:
        return 0.0
    distances = [
        float(np.linalg.norm(matrix[i] - matrix[j]))
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(distances)) if distances else 0.0


def run_summary(run_dir: str) -> dict[str, float | int]:
    """Summarize one realized world run without treating its agents as replicates."""
    features = behavioral_features_run(run_dir)
    agent_ids, matrix = feature_matrix(features)
    if len(agent_ids) == 0:
        raise ValueError(f"{run_dir!r} contains no analyzable agent records")

    summary: dict[str, float | int] = {
        "n_agents": len(agent_ids),
        "behavioral_dispersion": _mean_pairwise_distance(matrix),
    }
    means = matrix.mean(axis=0)
    summary.update({
        f"mean_{feature}": float(means[index])
        for index, feature in enumerate(FEATURE_NAMES)
    })
    # Run-level completion status as a reported field, not a silent filter
    # (this build's Phase 5). None means no status.json -- a legacy run or a
    # hand-built fixture, not evidence the run failed.
    summary["run_status"] = load_run_provenance(run_dir).get("status")
    return summary


def condition_attrition(sweep_dir: str) -> dict[str, object]:
    """Per-condition completion/attrition, read directly from each seed's
    provenance (never via behavioral analysis, which refuses a non-completed
    run outright -- see aiworld.analysis.provenance).

    Reporting this is the point: abort/invalid-output rate can correlate
    with condition (e.g. opaque vocabulary, long factual ledgers), and
    silently dropping those seeds from ``condition_summaries`` would bias
    exactly the crisis states a behavioral-commitment claim depends on.
    """
    run_dirs = seed_run_dirs(sweep_dir)
    statuses = {seed: load_run_provenance(path).get("status")
               for seed, path in run_dirs.items()}
    n_total = len(statuses)

    def _rate(status: str) -> float:
        return (sum(1 for s in statuses.values() if s == status) / n_total
               if n_total else 0.0)

    return {
        "n_seeds_total": n_total,
        "n_completed": sum(1 for s in statuses.values() if s == "completed"),
        "n_aborted": sum(1 for s in statuses.values() if s == "aborted"),
        "n_invalid_output": sum(1 for s in statuses.values() if s == "invalid_output"),
        "n_status_unknown": sum(1 for s in statuses.values() if s is None),
        "completed_rate": _rate("completed"),
        "aborted_rate": _rate("aborted"),
        "invalid_output_rate": _rate("invalid_output"),
        "seed_statuses": statuses,
    }


def seed_run_dirs(sweep_dir: str) -> dict[str, str]:
    """Return direct ``seed_*`` children keyed by their immutable seed directory."""
    paths = sorted(glob.glob(os.path.join(sweep_dir, "seed_*")))
    return {
        os.path.basename(path): path
        for path in paths
        if os.path.isdir(path)
    }


def condition_summaries(sweep_dir: str) -> dict[str, dict[str, float | int]]:
    """Summarize every completed world run immediately under one condition path."""
    return {seed: run_summary(path) for seed, path in seed_run_dirs(sweep_dir).items()}


def _bootstrap_mean_ci(differences: np.ndarray, *, iterations: int,
                       rng: np.random.Generator) -> tuple[float, float]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be at least 1")
    samples = rng.choice(differences, size=(iterations, len(differences)), replace=True)
    estimates = samples.mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _paired_sign_permutation_pvalue(differences: np.ndarray, *, iterations: int,
                                    rng: np.random.Generator) -> float:
    """Two-sided paired randomization estimate under sign exchangeability.

    This is a Monte Carlo test, reported as an exploratory p-value.  It is valid
    only for a contrast that was paired and specified before inspecting outcomes.
    The +1 correction prevents a spurious zero p-value from finite simulation.
    """
    if iterations < 1:
        raise ValueError("permutation iterations must be at least 1")
    observed = abs(float(differences.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(differences)))
    null_means = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(null_means >= observed) + 1) / (iterations + 1))


def paired_condition_comparison(
    left_sweep_dir: str,
    right_sweep_dir: str,
    *,
    metric: str = "behavioral_dispersion",
    bootstrap_iterations: int = 10_000,
    permutation_iterations: int = 10_000,
    random_state: int = 0,
) -> dict:
    """Compare a planned run-level outcome across matched world seeds.

    ``left - right`` is the reported effect. Only seed directories present in both
    sweeps are used; one-sided missing or failed runs are surfaced rather than
    silently substituted. This function is intentionally paired: unpaired runs
    require a different randomization design and should not borrow this test.
    """
    if metric not in RUN_LEVEL_METRICS:
        choices = ", ".join(RUN_LEVEL_METRICS)
        raise ValueError(f"Unknown run-level metric {metric!r}; choose one of: {choices}")

    left_run_dirs = seed_run_dirs(left_sweep_dir)
    right_run_dirs = seed_run_dirs(right_sweep_dir)
    left = condition_summaries(left_sweep_dir)
    right = condition_summaries(right_sweep_dir)
    paired_seeds = sorted(set(left) & set(right))
    if len(paired_seeds) < 2:
        raise ValueError(
            "Need at least two matched seed_* world runs for a paired condition "
            f"comparison; found {len(paired_seeds)}."
        )
    # One representative matched seed is enough: a sweep is one condition run
    # across seeds, so its provenance (model/decode/map/start mode) should be
    # uniform within the sweep by construction. This catches the realistic
    # failure mode -- comparing two different experiments/models entirely --
    # without re-reading every seed's metadata on every comparison.
    require_compatible_pair(
        left_run_dirs[paired_seeds[0]], right_run_dirs[paired_seeds[0]])

    left_values = np.array([float(left[seed][metric]) for seed in paired_seeds])
    right_values = np.array([float(right[seed][metric]) for seed in paired_seeds])
    differences = left_values - right_values
    rng = np.random.default_rng(random_state)
    ci_low, ci_high = _bootstrap_mean_ci(
        differences, iterations=bootstrap_iterations, rng=rng)
    p_value = _paired_sign_permutation_pvalue(
        differences, iterations=permutation_iterations, rng=rng)

    return {
        "metric": metric,
        "effect_definition": "left_minus_right",
        "n_matched_world_runs": len(paired_seeds),
        "matched_seeds": paired_seeds,
        "left_only_seeds": sorted(set(left) - set(right)),
        "right_only_seeds": sorted(set(right) - set(left)),
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": [ci_low, ci_high],
        "paired_sign_permutation_pvalue": p_value,
        "bootstrap_iterations": bootstrap_iterations,
        "permutation_iterations": permutation_iterations,
        "note": (
            "World runs are the replication unit; agents remain nested within a run. "
            "This is an exploratory paired estimate. Pre-specify the metric and "
            "contrast before inspecting outcomes, report failed/missing seeds, and "
            "do not interpret it as evidence of intrinsic purpose."
        ),
    }


def summarize_metrics(summaries: Iterable[dict[str, float | int]], metric: str) -> list[float]:
    """Small public helper for downstream scripts that need one chosen outcome."""
    if metric not in RUN_LEVEL_METRICS:
        raise ValueError(f"Unknown run-level metric {metric!r}")
    return [float(summary[metric]) for summary in summaries]
