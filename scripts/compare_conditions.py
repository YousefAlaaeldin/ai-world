"""Compare one pre-specified run-level outcome across paired condition sweeps.

Example::

    python scripts/compare_conditions.py \
      --left runs/factual_memory_unassigned/llm-flat-... \
      --right runs/deprimed_baseline/llm-flat-... \
      --metric behavioral_dispersion

This is deliberately separate from clustering. It uses world runs as the
replication unit and exposes any unmatched seed directories instead of treating the
agents inside a successful run as replacement samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.run_level import (  # noqa: E402
    RUN_LEVEL_METRICS,
    paired_condition_comparison,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True,
                    help="condition directory containing direct seed_* run directories")
    ap.add_argument("--right", required=True,
                    help="paired comparison condition directory")
    ap.add_argument("--metric", choices=RUN_LEVEL_METRICS,
                    default="behavioral_dispersion",
                    help="pre-specified run-level outcome to compare")
    ap.add_argument("--bootstrap-iterations", type=int, default=10_000)
    ap.add_argument("--permutation-iterations", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0,
                    help="analysis RNG seed for reproducible resampling")
    args = ap.parse_args()

    result = paired_condition_comparison(
        args.left,
        args.right,
        metric=args.metric,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        random_state=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
