"""Behavioral analysis CLI (Tranche 3, docs/REDESIGN_PLAN.md).

Unlike scripts/analyze_run.py (which analyzes self-model TEXT), this measures what
agents DO -- the primary outcome for the "emergent behavioral organization" claim,
and the only thing the de-primed baseline produces.

    # One run: per-agent behavioral fingerprint + behavioral divergence over time
    python scripts/analyze_behavior.py --run runs/live01

    # A sweep: cluster whole-run behavior across seeds, k chosen by silhouette
    python scripts/analyze_behavior.py --sweep runs/minimal_divergence

Zero LLM cost -- reads ticks.jsonl directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.behavioral_metrics import (  # noqa: E402
    behavioral_features_run, behavioral_divergence_over_time, cluster_behaviors,
    FEATURE_NAMES)


def _seed_dirs(sweep_dir: str) -> list[str]:
    dirs = sorted(glob.glob(os.path.join(sweep_dir, "seed_*")))
    return [d for d in dirs if os.path.isdir(d)]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="a single run dir (runs/<run_id>)")
    g.add_argument("--sweep", help="a sweep dir containing seed_* run dirs")
    ap.add_argument("--window", type=int, default=5,
                    help="block size for behavioral divergence over time (--run)")
    args = ap.parse_args()

    if args.run:
        feats = behavioral_features_run(args.run)
        if not feats:
            print(f"No ticks found in {args.run}")
            return
        print(f"Behavioral features — {args.run}")
        print("agent      " + "  ".join(f"{f[:10]:>10}" for f in FEATURE_NAMES))
        for aid in sorted(feats):
            row = "  ".join(f"{feats[aid][f]:>10.3f}" for f in FEATURE_NAMES)
            print(f"{aid:<10} {row}")
        print("\nBehavioral divergence over time "
              f"(window={args.window}, rising = differentiation):")
        div = behavioral_divergence_over_time(args.run, window=args.window)
        for tick in sorted(div):
            print(f"  tick {tick:>4}: {div[tick]:.4f}")
        return

    dirs = _seed_dirs(args.sweep)
    if not dirs:
        print(f"No seed_* run dirs under {args.sweep}")
        return
    print(f"Clustering behavior across {len(dirs)} runs under {args.sweep}")
    result = cluster_behaviors(dirs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
