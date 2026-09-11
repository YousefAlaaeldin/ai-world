"""Run the same experiment condition across multiple seeds.

DESIGN.md §6 is explicit: "n=1 means nothing with stochastic agents." A single
run is one data point about one particular sampling trajectory, not evidence
about whether environment/experience shapes purpose. This runs every seed in a
config's run.seeds list (or an override) by invoking run_stub.py / run_llm.py
once per seed, writing each to its own run_id under
runs/<experiment_name>/seed_<k>/ -- giving the analysis layer
(aiworld/analysis/trajectories.py) something to actually aggregate across.

Usage:
    # No-LLM mechanics sweep, zero cost, sanity-checks the whole seed range:
    python scripts/run_sweep.py --config configs/experiment_minimal.yaml --policy stub

    # Real sweep against the local model. One agent-tick = one LLM call (plus
    # one more on reflect ticks) x n_agents x ticks x n_seeds -- check that
    # product before running this unattended:
    python scripts/run_sweep.py --config configs/experiment_minimal.yaml --policy llm

    # The environment-contrast pair (DESIGN.md §6), same seeds so they're
    # directly comparable to the run above:
    python scripts/run_sweep.py --config configs/experiment_null_baseline.yaml --policy llm

    # See the commands without running anything:
    python scripts/run_sweep.py --config configs/experiment_minimal.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.config import load_config  # noqa: E402

SCRIPT_DIR = os.path.dirname(__file__)


def log(msg: str = "") -> None:
    """print() that's flushed immediately, so parent-process progress lines
    interleave correctly with each subprocess's inherited stdout instead of
    getting buffered and appearing out of order."""
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_minimal.yaml")
    ap.add_argument("--policy", choices=["stub", "llm"], default="stub")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override config run.seeds, e.g. --seeds 0 1 2")
    ap.add_argument("--prompt", choices=["flat", "leading"], default=None,
                    help="passed through to run_llm.py; ignored for --policy stub")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["run"]["seeds"]
    experiment_name = cfg.get(
        "experiment_name", os.path.splitext(os.path.basename(args.config))[0])
    script = "run_stub.py" if args.policy == "stub" else "run_llm.py"

    log(f"Sweeping '{experiment_name}' ({args.policy}) across "
          f"{len(seeds)} seeds: {seeds}")

    for seed in seeds:
        run_id = f"{experiment_name}/seed_{seed}"
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, script),
              "--config", args.config, "--seed", str(seed), "--run-id", run_id]
        if args.policy == "llm" and args.prompt:
            cmd += ["--prompt", args.prompt]

        log(f"\n--- seed {seed} ---")
        log(" ".join(cmd))
        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            log(f"seed {seed} FAILED (exit {result.returncode}) -- "
                  f"stopping sweep so a partial/broken condition isn't mistaken "
                  f"for a complete one.")
            sys.exit(result.returncode)

    if not args.dry_run:
        log(f"\nSweep complete. Logs in runs/{experiment_name}/seed_*/")


if __name__ == "__main__":
    main()
