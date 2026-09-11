"""Compute self-model divergence + purpose clusters from logged run output.

The actual measurement instrument the experiment needs (DESIGN.md §5) -- without
this, "did the environment affect purpose" can only be answered by reading
self-model text by eye. Uses aiworld.analysis.trajectories.tfidf_embed_fn by
default: a local, zero-cost, zero-network embedding (scikit-learn is already a
required dependency), since a "quick offline pass" is exactly what
trajectories.py's own docstring asks for. Swap in a real embedding model later
if paraphrase-level nuance starts to matter.

Usage:
    # Divergence over time for one run (needs self_models.jsonl -- stub runs
    # never write any, since nothing ever revises the self-model without an LLM):
    python scripts/analyze_run.py --run runs/live01

    # Purpose-cluster the FINAL self-model of every seed in a sweep -- recurring
    # clusters across independent seeds is DESIGN.md's headline finding:
    python scripts/analyze_run.py --sweep runs/minimal_divergence --clusters 5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.trajectories import (  # noqa: E402
    divergence_over_time, cluster_final_purposes, load_self_models, tfidf_embed_fn)


def analyze_run(run_dir: str) -> None:
    records = load_self_models(run_dir)
    if not records:
        print(f"No self_models.jsonl entries in {run_dir} -- stub runs never "
              f"write any (nothing revises the self-model without an LLM); "
              f"this needs a run from scripts/run_llm.py.")
        return

    div = divergence_over_time(run_dir, tfidf_embed_fn)
    if not div:
        print(f"Only one agent's self-model logged in {run_dir} -- divergence "
              f"needs at least two agents to compare.")
        return

    print(f"Divergence over time for {run_dir} "
          f"({len(records)} self-model versions logged):")
    for tick in sorted(div):
        print(f"  tick {tick:>4}: mean pairwise distance = {div[tick]:.4f}")

    ticks = sorted(div)
    if len(ticks) >= 2:
        trend = div[ticks[-1]] - div[ticks[0]]
        direction = "rising (differentiating)" if trend > 0 else "falling (converging)"
        print(f"  trend: {direction} ({trend:+.4f} from tick {ticks[0]} to {ticks[-1]})")
    if div.get(ticks[0], 0) > 0.01:
        print(f"  NOTE: tick {ticks[0]} distance is already > 0 -- agents may "
              f"not have started from a truly identical self-model. Expected "
              f"~0.0 at tick 0 if SelfModel.initial() was used for everyone.")


def analyze_sweep(sweep_dir: str, n_clusters: int) -> None:
    seed_dirs = sorted(glob.glob(os.path.join(sweep_dir, "seed_*")))
    if not seed_dirs:
        print(f"No seed_*/ subdirectories under {sweep_dir} -- run "
              f"scripts/run_sweep.py against this config first.")
        return

    result = cluster_final_purposes(seed_dirs, tfidf_embed_fn, n_clusters=n_clusters)
    if "error" in result:
        print(result["error"])
        return

    print(f"Clustered {result['n_final_models']} final self-models across "
          f"{len(seed_dirs)} seeds into {result['n_clusters']} clusters:")
    for cluster_id, members in sorted(result["clusters"].items()):
        distinct_seeds = sorted({m["run_id"] for m in members})
        print(f"\n  cluster {cluster_id}: {len(members)} agents from "
              f"{len(distinct_seeds)} distinct seeds "
              f"{'(recurs across seeds -- candidate purpose attractor)' if len(distinct_seeds) > 1 else '(single seed only)'}")
        for m in members[:5]:
            print(f"    [{m['run_id']}/{m['agent_id']}] {m['text']}")
        if len(members) > 5:
            print(f"    ... and {len(members) - 5} more")
    print("\nHand-label each cluster (DESIGN.md section 7: always hand-verify a sample "
          "of automated tags) -- this only computes membership, not names.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="single run directory, e.g. runs/live01")
    ap.add_argument("--sweep",
                    help="sweep directory containing seed_*/ subdirs, "
                    "e.g. runs/minimal_divergence")
    ap.add_argument("--clusters", type=int, default=5)
    args = ap.parse_args()

    if not args.run and not args.sweep:
        ap.error("pass --run <dir> or --sweep <dir>")

    if args.run:
        analyze_run(args.run)
    if args.sweep:
        analyze_sweep(args.sweep, args.clusters)


if __name__ == "__main__":
    main()
