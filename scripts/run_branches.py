"""Run one world and generate branch perturbations at predeclared checkpoints.

Stub policy only in this build phase (no live LLM calls, matching the
project's stub/fake-policy-only constraint) -- both the main run and every
branch use aiworld.sim.stub_policy.make_stub_policy. Swapping in a real
model-backed policy for the focal agent is a small, isolated future change
(pass a different ``focal_policy_factory``); it is not wired to a CLI flag
here deliberately, since running one against a live model is a pilot-stage
decision, not an infrastructure one.

Usage (mechanics only, zero cost):
    python scripts/run_branches.py --config configs/experiment_factual_memory_unassigned.yaml \\
        --probe configs/probes/probe_01_retain_vs_transfer.yaml \\
        --focal-agent agent_0 --run-id branch_smoke01 --max-calls 10000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.config import load_config  # noqa: E402
from aiworld.experiments.branches import (  # noqa: E402
    BRANCH_TYPES,
    enforce_call_budget,
    eligible_now,
    generate_branches,
    predeclare_checkpoints_fixed,
    run_branch,
)
from aiworld.experiments.specs import ProbeSpec, load_probe_spec  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.provenance import (  # noqa: E402
    ACTION_API_VERSION, ANALYSIS_SCHEMA_VERSION, git_revision)
from aiworld.seed_plan import SeedPlan  # noqa: E402
from aiworld.sim.stub_policy import make_stub_policy  # noqa: E402
from scripts.run_stub import build_world  # noqa: E402


def run_branches_for_condition(
    cfg: dict,
    run_id: str,
    *,
    probe: ProbeSpec,
    focal_agent_id: str,
    seed: int = 0,
    checkpoint_mode: str = "fixed",
    checkpoint_every_n: int = 10,
    branch_horizon: int = 5,
    branch_types: tuple[str, ...] = BRANCH_TYPES,
    branch_mode: str = "microprobe",
    max_calls: int | None = None,
    logs_root: str = "runs",
    focal_policy_factory=None,
) -> dict:
    """Build one world, step it, and generate every requested branch type at
    every predeclared checkpoint.

    ``checkpoint_mode='fixed'`` fires at multiples of ``checkpoint_every_n``
    (an exact, pre-computable schedule). ``checkpoint_mode='eligible'`` fires
    whenever the focal agent's LIVE decision state satisfies
    ``probe.eligible_state_predicate`` (aiworld.experiments.branches.eligible_now)
    -- checked before any branch exists, so it can never be "selected because
    it looked purposeful" after the fact. Call budget is estimated and
    enforced BEFORE the run starts, using an exact count (fixed mode) or a
    conservative upper bound of one checkpoint per tick (eligible mode).
    """
    if checkpoint_mode not in ("fixed", "eligible"):
        raise ValueError(f"Unknown checkpoint_mode {checkpoint_mode!r}; "
                         "expected 'fixed' or 'eligible'.")

    n_agents = cfg["agents"]["count"]
    mortality = cfg["world"]["mortality"]
    ticks_total = cfg["run"]["ticks"]
    focal_policy_factory = focal_policy_factory or (
        lambda run_seed: make_stub_policy(random.Random(run_seed)))

    if checkpoint_mode == "fixed":
        fixed_checkpoints = set(predeclare_checkpoints_fixed(ticks_total, checkpoint_every_n))
        n_checkpoints_budgeted = len(fixed_checkpoints)
    else:
        fixed_checkpoints = None
        # Conservative upper bound: every tick could in principle fire.
        n_checkpoints_budgeted = ticks_total

    n_focal_agents_per_branch = n_agents if branch_mode == "ecological" else 1
    budget = enforce_call_budget(
        n_checkpoints_budgeted, len(branch_types), branch_horizon,
        n_focal_agents_per_branch, max_calls)

    world, setup = build_world(cfg, run_id, n_agents, seed, mortality=mortality,
                               root=logs_root)
    world.logger.log_meta({
        "run_id": run_id,
        "policy": "stub",
        "seed": seed,
        "mortality": mortality,
        "n_agents": n_agents,
        "action_api_version": ACTION_API_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "branch_run": {
            "probe_id": probe.id,
            "focal_agent_id": focal_agent_id,
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_every_n": checkpoint_every_n if checkpoint_mode == "fixed" else None,
            "branch_horizon": branch_horizon,
            "branch_types": list(branch_types),
            "branch_mode": branch_mode,
            "estimated_call_budget": budget,
            "max_calls": max_calls,
        },
        **setup,
    })

    seed_plan = SeedPlan(run_seed=seed)
    main_policy = focal_policy_factory(seed)
    branch_manifests: list[dict] = []
    status, error = "completed", None

    try:
        for _ in range(ticks_total):
            focal_agent = next(
                (a for a in world.agents
                 if a.agent_id == focal_agent_id and a.drives.alive), None)
            fires = False
            if focal_agent is not None:
                upcoming_tick = world.tick_count + 1
                if checkpoint_mode == "fixed":
                    fires = upcoming_tick in fixed_checkpoints
                else:
                    fires = eligible_now(probe.eligible_state_predicate, world, focal_agent)

            if fires:
                pre_snapshot = world.snapshot()
                checkpoint_tick = pre_snapshot.tick_count + 1
                for manifest, branched_snapshot in generate_branches(
                        run_id, pre_snapshot, checkpoint_tick, focal_agent_id,
                        seed_plan, branch_types=branch_types, mode=branch_mode):
                    branch_dir = f"{run_id}/branches/{manifest.branch_id}"
                    branch_logger = RunLogger(branch_dir, root=logs_root)
                    run_branch(
                        manifest, branched_snapshot, branch_horizon, branch_logger,
                        focal_policy=focal_policy_factory(manifest.branch_seed))
                    branch_logger.finalize(
                        status="completed",
                        action_api_version=ACTION_API_VERSION,
                        analysis_schema_version=ANALYSIS_SCHEMA_VERSION)
                    manifest_path = os.path.join(logs_root, branch_dir, "manifest.json")
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest.to_dict(), f, indent=2)
                    branch_manifests.append(manifest.to_dict())

            world.step(main_policy, reflect_every=0, reflect_fn=None)
    except BaseException as exc:  # noqa: BLE001
        # BaseException, not Exception: see scripts/run_llm.py's matching comment --
        # KeyboardInterrupt must also be labeled "aborted", not "completed".
        status, error = "aborted", str(exc)
        raise
    finally:
        world.logger.finalize(
            status=status, expected_ticks=ticks_total,
            observed_ticks=world.tick_count, action_api_version=ACTION_API_VERSION,
            analysis_schema_version=ANALYSIS_SCHEMA_VERSION,
            git_revision=git_revision(), error=error)

    return {
        "run_id": run_id,
        "estimated_call_budget": budget,
        "n_branches": len(branch_manifests),
        "branches": branch_manifests,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment_factual_memory_unassigned.yaml")
    ap.add_argument("--probe", required=True, help="path to a probe spec YAML")
    ap.add_argument("--focal-agent", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default="branches01")
    ap.add_argument("--checkpoint-mode", choices=["fixed", "eligible"], default="fixed")
    ap.add_argument("--checkpoint-every-n", type=int, default=10)
    ap.add_argument("--branch-horizon", type=int, default=5)
    ap.add_argument("--branch-types", nargs="+", default=list(BRANCH_TYPES),
                    choices=list(BRANCH_TYPES))
    ap.add_argument("--branch-mode", choices=["microprobe", "ecological"], default="microprobe")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="refuse to run if the estimated call budget exceeds this")
    args = ap.parse_args()

    cfg = load_config(args.config)
    probe = load_probe_spec(args.probe)
    if probe.status != "active":
        raise ValueError(f"probe {probe.id!r} has status {probe.status!r}, not 'active'.")

    summary = run_branches_for_condition(
        cfg, args.run_id, probe=probe, focal_agent_id=args.focal_agent,
        seed=args.seed, checkpoint_mode=args.checkpoint_mode,
        checkpoint_every_n=args.checkpoint_every_n,
        branch_horizon=args.branch_horizon, branch_types=tuple(args.branch_types),
        branch_mode=args.branch_mode, max_calls=args.max_calls)

    print(f"Run '{args.run_id}' done. {summary['n_branches']} branches generated "
         f"(estimated budget {summary['estimated_call_budget']}).")
    print(f"Logs in runs/{args.run_id}/ and runs/{args.run_id}/branches/")


if __name__ == "__main__":
    main()
