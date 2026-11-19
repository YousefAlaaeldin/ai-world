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
import copy
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem  # noqa: E402
from aiworld.self_model import SelfModelHistory, SHARED_DIRECTIVE  # noqa: E402
from aiworld.logging_schema import RunLogger, CompletionRecord  # noqa: E402
from aiworld.sim.world import World, Agent, agent_recent_memories  # noqa: E402
from aiworld.llm_client import LLMClient, config_from_env  # noqa: E402
from aiworld.seed_plan import SeedPlan  # noqa: E402
from aiworld.provenance import (  # noqa: E402
    ACTION_API_VERSION, ANALYSIS_SCHEMA_VERSION, git_revision)
from aiworld.model_spec import DEFAULT_MODEL_REGISTRY_PATH, load_model_spec  # noqa: E402
from aiworld.components.llm_policy import (  # noqa: E402
    ACTION_MEMORY_FACTUAL_LEDGER, ACTION_MEMORY_MODES,
    DEFAULT_ACTION_CONTEXT_MAX_CHARS, DEFAULT_ACTION_MEMORY_MAX_EVENTS,
    make_llm_policy, _action_system_prompt)
from aiworld.components.self_model_component import (  # noqa: E402
    make_reflect_fn, FLAT_REVISION_PROMPT, LEADING_REVISION_PROMPT,
    NON_IDENTITY_SUMMARY_PROMPT, parse_non_identity_summary_diagnostic)
from aiworld.config import (  # noqa: E402
    load_config, drive_specs_from_config, initial_locations, adjacency_from_config,
    resource_map_from_config)


def build_world(cfg, run_id, n_agents, seed, mortality, vocabulary="neutral",
                memory_style="narrative", initial_directive=SHARED_DIRECTIVE):
    """Construct a world plus the non-secret provenance known at setup time."""
    # Keep map/start allocation independent of action conflict resolution.  A
    # later change in one setup draw must not silently alter the random-priority
    # sequence that resolves otherwise-identical action proposals.
    seed_plan = SeedPlan(run_seed=seed)
    map_seed = seed_plan.seed_for("resource_map")
    start_seed = seed_plan.seed_for("initial_locations")
    world_seed = seed_plan.seed_for("presentation_and_resolution")
    map_rng = random.Random(map_seed)
    start_rng = random.Random(start_seed)
    world_rng = random.Random(world_seed)
    locations = cfg["world"]["locations"]
    specs = drive_specs_from_config(cfg)
    adjacency = adjacency_from_config(cfg)
    map_mode = cfg["world"].get("resource_map_mode", "fixed")
    resources, resource_map = resource_map_from_config(
        cfg, rng=map_rng, map_mode=map_mode)
    start_mode = cfg.get("agents", {}).get("start_mode", "legacy")
    start_locations = initial_locations(start_rng, locations, n_agents, mode=start_mode)
    agents = [
        Agent(agent_id=f"agent_{i}", location=start_locations[i],
              drives=DriveSystem(specs=specs, vocabulary=vocabulary),
              history=SelfModelHistory(
                  f"agent_{i}", initial_directive=initial_directive))
        for i in range(n_agents)
    ]
    logger = RunLogger(run_id)
    world = World(
        locations, resources, agents, logger, rng=world_rng, mortality=mortality,
        adjacency=adjacency, vocabulary=vocabulary, memory_style=memory_style,
        communication_mode=cfg.get("communication", {}).get("mode", "none"),
        crisis_reflection_enabled=cfg.get("reflection", {}).get(
            "crisis_triggered", False),
        randomize_opaque_labels=cfg["world"].get(
            "randomize_opaque_labels", True),
        adjacent_resource_preview=cfg["world"].get(
            "adjacent_resource_preview", False),
    )
    setup = {
        "initial_locations": {
            agent.agent_id: agent.location for agent in agents
        },
        "initial_location_mode": start_mode,
        "resource_map": resource_map,
        "resource_map_mode": map_mode,
        "presentation": world.presentation_metadata(),
        "action_resolution": "simultaneous_proposals_random_priority",
        "randomization_seeds": {
            "resource_map": map_seed,
            "initial_locations": start_seed,
            "presentation_and_resolution": world_seed,
        },
    }
    return world, setup


def config_digest(cfg: dict) -> str:
    """Stable identifier for the fully resolved, JSON-safe experimental config."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_minimal.yaml")
    ap.add_argument("--agents", type=int, default=None, help="override config agents.count")
    ap.add_argument("--ticks", type=int, default=None, help="override config run.ticks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampling-seed", type=int, default=None,
                    help="LLM sampling seed; defaults to AIWORLD_SAMPLING_SEED or --seed")
    ap.add_argument("--run-id", default="live01",
                    help="immutable output id; reusing one is an error")
    ap.add_argument("--reflect-every", type=int, default=None,
                    help="override config reflection.every_n_ticks")
    ap.add_argument("--prompt", choices=["flat", "leading"], default=None,
                    help="override config reflection.prompt")
    ap.add_argument("--vocabulary", choices=["neutral", "biological", "opaque"],
                    default=None,
                    help="override config world.vocabulary (neutral strips biological "
                         "priors; biological is the contrast arm; opaque is fully "
                         "symbolic S1/R1 framing, no words carrying meaning)")
    ap.add_argument("--memory-style", choices=["narrative", "factual"], default=None,
                    help="override config world.memory_style (narrative is the "
                         "curated-framing default; factual is a plain event ledger "
                         "with no 'nearly died'/'rescued'-style narrative loading)")
    ap.add_argument("--immortal", action="store_true")
    ap.add_argument("--model-spec", default=None,
                    help="name of a ModelSpec in --model-registry to record in "
                         "provenance (aiworld/model_spec.py); optional, purely "
                         "additive metadata -- omit to keep today's behavior")
    ap.add_argument("--model-registry", default=DEFAULT_MODEL_REGISTRY_PATH,
                    help="path to the ModelSpec registry YAML")
    args = ap.parse_args()

    exp_cfg = load_config(args.config)
    n_agents = args.agents if args.agents is not None else exp_cfg["agents"]["count"]
    ticks = args.ticks if args.ticks is not None else exp_cfg["run"]["ticks"]
    reflect_every = (args.reflect_every if args.reflect_every is not None
                     else exp_cfg["reflection"]["every_n_ticks"])
    prompt_choice = args.prompt or exp_cfg["reflection"]["prompt"]
    vocabulary = (args.vocabulary
                  or exp_cfg["world"].get("vocabulary", "neutral"))
    memory_style = (args.memory_style
                    or exp_cfg["world"].get("memory_style", "narrative"))
    mortality = exp_cfg["world"]["mortality"] and not args.immortal

    # Condition arm (docs/REDESIGN_PLAN.md). The de-primed baseline has no
    # directive and no self-model feedback; the positive control can turn them
    # back on explicitly in YAML.
    agents_cfg = exp_cfg.get("agents", {})
    show_self_model = agents_cfg.get("show_self_model", True)
    directive = agents_cfg.get("directive", SHARED_DIRECTIVE)  # YAML null -> de-primed
    action_memory = agents_cfg.get("action_memory", "none")
    action_memory_max_events = agents_cfg.get(
        "action_memory_max_events", DEFAULT_ACTION_MEMORY_MAX_EVENTS)
    action_context_max_chars = agents_cfg.get(
        "action_context_max_chars", DEFAULT_ACTION_CONTEXT_MAX_CHARS)
    # Phase 7 non-identity summary control: same extra reflection call and
    # context length as the self-narrative arm, but neutral factual content
    # (aiworld.components.self_model_component.NON_IDENTITY_SUMMARY_PROMPT)
    # instead of "Who I am / What matters / What I want now".
    self_narrative_control = agents_cfg.get("self_narrative_control", False)
    reflection_enabled = exp_cfg["reflection"].get("enabled", True)

    if action_memory not in ACTION_MEMORY_MODES:
        choices = ", ".join(sorted(ACTION_MEMORY_MODES))
        raise ValueError(
            f"Unknown agents.action_memory {action_memory!r}; expected one of: {choices}."
        )
    if not isinstance(action_memory_max_events, int) or action_memory_max_events < 1:
        raise ValueError("agents.action_memory_max_events must be an integer >= 1.")
    if not isinstance(action_context_max_chars, int) or action_context_max_chars < 96:
        raise ValueError("agents.action_context_max_chars must be an integer >= 96.")
    # A factual ledger is an experimental history channel, not merely a label.
    # Reject the narratively phrased world-memory source so the condition cannot
    # silently acquire "nearly died"/identity-like framing.
    if (action_memory == ACTION_MEMORY_FACTUAL_LEDGER
            and memory_style != "factual"):
        raise ValueError(
            "agents.action_memory: factual_ledger requires "
            "world.memory_style: factual."
        )

    cfg = config_from_env()
    sampling_seed = (args.sampling_seed if args.sampling_seed is not None
                     else cfg.sampling_seed if cfg.sampling_seed is not None
                     else args.seed)
    cfg.sampling_seed = sampling_seed
    client = LLMClient(cfg)
    print(f"Using model '{cfg.model}' at {cfg.base_url} (provider={cfg.provider})")

    # Persist the actual experimental condition, including one-off CLI overrides,
    # rather than a config file that only approximately describes the run.
    resolved_cfg = copy.deepcopy(exp_cfg)
    resolved_agents = resolved_cfg.setdefault("agents", {})
    resolved_agents["count"] = n_agents
    # Persist every condition knob even when an older config relied on a legacy
    # default.  A rerun therefore cannot be mistaken for an unprimed arm simply
    # because its source YAML omitted a field.
    resolved_agents["show_self_model"] = show_self_model
    resolved_agents["directive"] = directive
    resolved_agents["action_memory"] = action_memory
    resolved_agents["action_memory_max_events"] = action_memory_max_events
    resolved_agents["action_context_max_chars"] = action_context_max_chars
    resolved_agents["self_narrative_control"] = self_narrative_control
    resolved_cfg.setdefault("world", {})["mortality"] = mortality
    resolved_cfg["world"]["vocabulary"] = vocabulary
    resolved_cfg["world"]["memory_style"] = memory_style
    resolved_cfg.setdefault("reflection", {})["every_n_ticks"] = reflect_every
    resolved_cfg["reflection"]["prompt"] = prompt_choice
    resolved_cfg.setdefault("run", {})["ticks"] = ticks
    resolved_cfg["run"]["seed"] = args.seed
    resolved_cfg["run"]["sampling_seed"] = sampling_seed
    resolved_cfg["run"]["model"] = cfg.model

    action_system_prompt = _action_system_prompt(
        vocabulary, directive, show_self_model)
    reflection_template = (
        NON_IDENTITY_SUMMARY_PROMPT if self_narrative_control
        else FLAT_REVISION_PROMPT if prompt_choice == "flat"
        else LEADING_REVISION_PROMPT)
    world, setup = build_world(resolved_cfg, args.run_id, n_agents, args.seed,
                               mortality=mortality, vocabulary=vocabulary,
                               memory_style=memory_style,
                               initial_directive=directive)
    condition = {
        "show_self_model": show_self_model,
        "directive": directive,
        "initial_self_model_is_blank": directive is None or directive == "",
        "action_memory": action_memory,
        "action_memory_max_events": action_memory_max_events,
        "action_context_max_chars": action_context_max_chars,
        "self_narrative_control": self_narrative_control,
        "world_memory_style": memory_style,
        "vocabulary": vocabulary,
        "reflection_enabled": reflection_enabled,
        "reflection_prompt": ("non_identity_summary" if self_narrative_control
                              else prompt_choice if reflection_enabled else None),
        "communication_mode": world.communication_mode,
        "crisis_reflection_enabled": world.crisis_reflection_enabled,
    }
    world.logger.log_meta({
        "run_id": args.run_id,
        "policy": "llm",
        "seed": args.seed,
        "sampling_seed": sampling_seed,
        "mortality": mortality,
        "n_agents": n_agents,
        "condition": condition,
        "resolved_config": resolved_cfg,
        "resolved_config_sha256": config_digest(resolved_cfg),
        "prompts": {
            "action_system": action_system_prompt,
            "reflection_template": reflection_template if reflection_enabled else None,
        },
        "llm": client.request_metadata(),
        "action_api_version": ACTION_API_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "model_spec": (load_model_spec(args.model_spec, path=args.model_registry).to_dict()
                      if args.model_spec else None),
        **setup,
    })
    policy = make_llm_policy(client, vocabulary=vocabulary, directive=directive,
                             show_self_model=show_self_model,
                             action_memory=action_memory,
                             action_memory_max_events=action_memory_max_events,
                             action_context_max_chars=action_context_max_chars,
                             self_narrative_control=self_narrative_control)

    reflect = None
    if reflection_enabled:
        history_by_agent = {a.agent_id: a.history for a in world.agents}

        def log_reflection_completion(record: dict) -> None:
            world.logger.log_completion(CompletionRecord(
                phase=record["phase"],
                tick=record["tick"],
                agent_id=record["agent_id"],
                system_prompt=record["system_prompt"],
                user_prompt=record["prompt"],
                raw_completion=record["raw_completion"],
                parsed_output=json.dumps(record["parsed_output"], ensure_ascii=False,
                                         sort_keys=True),
                parse_status=record["parse_status"],
                request_metadata=client.request_metadata(),
                response_metadata=client.response_metadata(),
            ))

        reflect = make_reflect_fn(
            model_call=lambda p: client.complete(p),
            history_by_agent=history_by_agent,
            logger=world.logger,
            prompt_template=reflection_template,
            recent_memories_fn=agent_recent_memories,
            completion_observer=log_reflection_completion,
            render_self_model_fn=(
                (lambda self_model: self_model.as_summary_prompt_block())
                if self_narrative_control else None),
            parse_fn=(parse_non_identity_summary_diagnostic
                     if self_narrative_control else None),
        )

    # Log every agent's identical starting self-model as version 0.
    for a in world.agents:
        world.logger.log_self_model(a.agent_id, a.history.current.to_dict())

    status, error = "completed", None
    try:
        world.run(policy, ticks=ticks,
                 reflect_every=reflect_every, reflect_fn=reflect)
    except BaseException as exc:
        # BaseException, not Exception: a Ctrl+C mid-run raises KeyboardInterrupt,
        # which does NOT subclass Exception. Catching only Exception here left an
        # interrupted, truncated run mislabeled "completed" in the immutable
        # status.json (require_completed_run has no independent tick-count check,
        # so it would have silently accepted a partial run as a full one). Always
        # re-raised immediately below, so process-exit behavior is unchanged.
        status, error = "aborted", str(exc)
        raise
    finally:
        world.logger.finalize(
            status=status,
            expected_ticks=ticks,
            observed_ticks=world.tick_count,
            action_api_version=ACTION_API_VERSION,
            analysis_schema_version=ANALYSIS_SCHEMA_VERSION,
            git_revision=git_revision(),
            error=error,
        )

    alive = sum(1 for a in world.agents if a.drives.alive)
    print(f"Run '{args.run_id}' done. {alive}/{n_agents} alive. "
          f"LLM stats: {client.stats()}")
    print(f"Logs in runs/{args.run_id}/")


if __name__ == "__main__":
    main()
