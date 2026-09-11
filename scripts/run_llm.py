"""Run a simulation with a real (or local) LLM backing both action and reflection.

Everything routes through one LLMClient, so this runs identically against:
  - a local model on your GPU (Ollama/vLLM, the default), or
  - a hosted API (set the AIWORLD_* env vars, see llm_client.config_from_env).

Usage (local Ollama, after `ollama serve` +
`ollama pull huihui_ai/mistral-small-abliterated:24b`):
    python scripts/run_llm.py --agents 6 --ticks 30 --run-id live01

Prompt template for reflection is chosen by --prompt {flat,leading}. Use 'flat' for
real runs; 'leading' only for the confound check.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.sim.world import World, Agent, agent_recent_memories  # noqa: E402
from aiworld.llm_client import LLMClient, config_from_env  # noqa: E402
from aiworld.components.llm_policy import make_llm_policy  # noqa: E402
from aiworld.components.self_model_component import (  # noqa: E402
    make_reflect_fn, FLAT_REVISION_PROMPT, LEADING_REVISION_PROMPT)
from aiworld.config import (  # noqa: E402
    load_config, drive_specs_from_config, resources_from_config, initial_locations,
    adjacency_from_config)


def build_world(cfg, run_id, n_agents, seed, mortality, model_name):
    rng = random.Random(seed)
    locations = cfg["world"]["locations"]
    resources = resources_from_config(cfg)
    specs = drive_specs_from_config(cfg)
    adjacency = adjacency_from_config(cfg)
    start_locations = initial_locations(rng, locations, n_agents)
    agents = [
        Agent(agent_id=f"agent_{i}", location=start_locations[i],
              drives=DriveSystem(specs=specs), history=SelfModelHistory(f"agent_{i}"))
        for i in range(n_agents)
    ]
    logger = RunLogger(run_id)
    logger.log_meta({
        "run_id": run_id, "n_agents": n_agents, "seed": seed,
        "mortality": mortality, "policy": "llm", "model": model_name,
    })
    return World(locations, resources, agents, logger, rng=rng, mortality=mortality,
                adjacency=adjacency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_minimal.yaml")
    ap.add_argument("--agents", type=int, default=None, help="override config agents.count")
    ap.add_argument("--ticks", type=int, default=None, help="override config run.ticks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default="live01")
    ap.add_argument("--reflect-every", type=int, default=None,
                    help="override config reflection.every_n_ticks")
    ap.add_argument("--prompt", choices=["flat", "leading"], default=None,
                    help="override config reflection.prompt")
    ap.add_argument("--immortal", action="store_true")
    args = ap.parse_args()

    exp_cfg = load_config(args.config)
    n_agents = args.agents if args.agents is not None else exp_cfg["agents"]["count"]
    ticks = args.ticks if args.ticks is not None else exp_cfg["run"]["ticks"]
    reflect_every = (args.reflect_every if args.reflect_every is not None
                     else exp_cfg["reflection"]["every_n_ticks"])
    prompt_choice = args.prompt or exp_cfg["reflection"]["prompt"]
    mortality = exp_cfg["world"]["mortality"] and not args.immortal

    cfg = config_from_env()
    client = LLMClient(cfg)
    print(f"Using model '{cfg.model}' at {cfg.base_url} (provider={cfg.provider})")

    world = build_world(exp_cfg, args.run_id, n_agents, args.seed,
                        mortality=mortality, model_name=cfg.model)
    policy = make_llm_policy(client)

    history_by_agent = {a.agent_id: a.history for a in world.agents}
    template = FLAT_REVISION_PROMPT if prompt_choice == "flat" else LEADING_REVISION_PROMPT
    reflect = make_reflect_fn(
        model_call=lambda p: client.complete(p),
        history_by_agent=history_by_agent,
        logger=world.logger,
        prompt_template=template,
        recent_memories_fn=agent_recent_memories,
    )

    # Log every agent's identical starting self-model as version 0.
    for a in world.agents:
        world.logger.log_self_model(a.agent_id, a.history.current.to_dict())

    world.run(policy, ticks=ticks,
              reflect_every=reflect_every, reflect_fn=reflect)
    world.logger.close()

    alive = sum(1 for a in world.agents if a.drives.alive)
    print(f"Run '{args.run_id}' done. {alive}/{n_agents} alive. "
          f"LLM stats: {client.stats()}")
    print(f"Logs in runs/{args.run_id}/")


if __name__ == "__main__":
    main()
