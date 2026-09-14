"""Run a full simulation with the stub (no-LLM) policy.

Proves the mechanics + logging end to end with zero API cost. Produces real
runs/<run_id>/*.jsonl files the analysis layer can chew on.

Usage:
    python scripts/run_stub.py --agents 6 --ticks 40 --run-id smoke01
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.sim.world import World, Agent  # noqa: E402
from aiworld.sim.stub_policy import make_stub_policy  # noqa: E402
from aiworld.config import (  # noqa: E402
    load_config, drive_specs_from_config, initial_locations, adjacency_from_config,
    resource_map_from_config)


def build_world(cfg: dict, run_id: str, n_agents: int, seed: int,
                mortality: bool) -> tuple[World, dict]:
    rng = random.Random(seed)
    locations = cfg["world"]["locations"]
    specs = drive_specs_from_config(cfg)
    adjacency = adjacency_from_config(cfg)
    vocabulary = cfg["world"].get("vocabulary", "neutral")
    memory_style = cfg["world"].get("memory_style", "narrative")
    map_mode = cfg["world"].get("resource_map_mode", "fixed")
    resources, resource_map = resource_map_from_config(
        cfg, rng=rng, map_mode=map_mode)
    start_mode = cfg.get("agents", {}).get("start_mode", "legacy")
    # All agents start identical except ID and a counterbalanced location.
    start_locations = initial_locations(rng, locations, n_agents, mode=start_mode)
    agents = [
        Agent(
            agent_id=f"agent_{i}",
            location=start_locations[i],
            drives=DriveSystem(specs=specs, vocabulary=vocabulary),
            history=SelfModelHistory(f"agent_{i}"),
        )
        for i in range(n_agents)
    ]
    logger = RunLogger(run_id)
    world = World(
        locations, resources, agents, logger,
        rng=rng, mortality=mortality, adjacency=adjacency,
        vocabulary=vocabulary, memory_style=memory_style,
        communication_mode=cfg.get("communication", {}).get("mode", "none"),
        crisis_reflection_enabled=cfg.get("reflection", {}).get(
            "crisis_triggered", False),
        randomize_opaque_labels=cfg["world"].get(
            "randomize_opaque_labels", True),
    )
    return world, {
        "initial_locations": {agent.agent_id: agent.location for agent in agents},
        "initial_location_mode": start_mode,
        "resource_map": resource_map,
        "resource_map_mode": map_mode,
        "presentation": world.presentation_metadata(),
        "action_resolution": "simultaneous_proposals_random_priority",
    }


def config_digest(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_minimal.yaml")
    ap.add_argument("--agents", type=int, default=None, help="override config agents.count")
    ap.add_argument("--ticks", type=int, default=None, help="override config run.ticks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default="smoke01",
                    help="immutable output id; reusing one is an error")
    ap.add_argument("--immortal", action="store_true", help="disable mortality")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_agents = args.agents if args.agents is not None else cfg["agents"]["count"]
    ticks = args.ticks if args.ticks is not None else cfg["run"]["ticks"]
    reflect_every = cfg["reflection"]["every_n_ticks"]
    mortality = cfg["world"]["mortality"] and not args.immortal
    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg.setdefault("agents", {})["count"] = n_agents
    resolved_cfg.setdefault("world", {})["mortality"] = mortality
    resolved_cfg.setdefault("run", {})["ticks"] = ticks
    resolved_cfg["run"]["seed"] = args.seed

    world, setup = build_world(
        resolved_cfg, args.run_id, n_agents, args.seed, mortality=mortality)
    world.logger.log_meta({
        "run_id": args.run_id,
        "policy": "stub",
        "seed": args.seed,
        "mortality": mortality,
        "n_agents": n_agents,
        "resolved_config": resolved_cfg,
        "resolved_config_sha256": config_digest(resolved_cfg),
        "condition": {
            "communication_mode": world.communication_mode,
            "crisis_reflection_enabled": world.crisis_reflection_enabled,
        },
        "note": "Stub policy — mechanics smoke test only, not a result.",
        **setup,
    })
    policy = make_stub_policy(random.Random(args.seed))
    world.run(policy, ticks=ticks, reflect_every=reflect_every, reflect_fn=None)
    world.logger.close()

    alive = sum(1 for a in world.agents if a.drives.alive)
    print(f"Run '{args.run_id}' complete: {ticks} ticks, "
          f"{alive}/{n_agents} agents alive at end.")
    print(f"Logs in runs/{args.run_id}/")


if __name__ == "__main__":
    main()
