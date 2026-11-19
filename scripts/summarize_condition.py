"""Aggregate run-level outcomes across every seed of ONE condition sweep.

Complements scripts/analyze_behavior.py --sweep (which clusters individual
agent-vectors) and scripts/compare_conditions.py (which needs a SECOND
condition to pair against). This is the missing single-condition step:
per-seed aiworld.analysis.run_level.run_summary(), aggregated into a mean/std
for each metric in RUN_LEVEL_METRICS -- most importantly behavioral_dispersion,
the pre-registered primary outcome (docs/ANALYSIS_PLAN_TEMPLATE.md).

This is also exactly the between-seed variance estimate docs/POWER_ANALYSIS.md
has been blocked on since the build: sigma_hat for behavioral_dispersion,
computed for real for the first time once a condition has enough completed
seeds.

Always reports condition_attrition first, not just for completed seeds --
an aborted/excluded seed is a validity fact, not something to silently drop.

Example::

    python scripts/summarize_condition.py --sweep runs/<experiment>/<condition>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from aiworld.analysis.run_level import (  # noqa: E402
    RUN_LEVEL_METRICS,
    condition_attrition,
    condition_summaries,
)


def aggregate(summaries: dict[str, dict]) -> dict[str, dict[str, float]]:
    """mean/std/min/max/n for every RUN_LEVEL_METRICS column, across seeds
    whose run_summary succeeded (already-excluded seeds never reach here --
    see condition_summaries' own require_completed_run guard)."""
    agg: dict[str, dict[str, float]] = {}
    for metric in RUN_LEVEL_METRICS:
        values = np.array([s[metric] for s in summaries.values() if metric in s],
                          dtype=float)
        if len(values) == 0:
            continue
        agg[metric] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    help="condition directory containing direct seed_* run directories "
                         "(NOT the experiment root -- one level deeper, same path you'd "
                         "pass to analyze_behavior.py --sweep)")
    ap.add_argument("--per-seed", action="store_true",
                    help="also print each seed's full run_summary, not just the aggregate")
    args = ap.parse_args()

    attrition = condition_attrition(args.sweep)
    print("=== condition_attrition ===")
    print(json.dumps(attrition, indent=2))

    summaries = condition_summaries(args.sweep)
    if not summaries:
        print("\nNo completed seeds to summarize.")
        return

    print(f"\n=== aggregate across {len(summaries)} completed seed(s) ===")
    print(json.dumps(aggregate(summaries), indent=2))

    if args.per_seed:
        print("\n=== per-seed run_summary ===")
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
