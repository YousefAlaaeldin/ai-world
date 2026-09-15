"""Run a static-context sampling null for an LLM experiment condition.

This is deliberately *not* another environment condition.  It freezes the exact
initial world/context from a normal YAML condition, repeatedly samples an action
from that one unchanged prompt, and never applies the action.  It therefore
estimates variation due to the model/decoder under a fixed context, rather than
mistaking a different topology, scarcity schedule, or start placement for a
"sampling null".

Use it alongside a real sweep with the same config and seeds::

    python scripts/run_sweep.py --config configs/experiment_factual_memory_unassigned.yaml --policy llm
    python scripts/run_sweep.py --config configs/experiment_factual_memory_unassigned.yaml --policy sampling-null

The output is proposal-only data.  It must never be mixed with realized behavior
from ``run_llm.py``: metadata and every tick record make that distinction explicit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.components.llm_policy import (  # noqa: E402
    ACTION_MEMORY_FACTUAL_LEDGER,
    ACTION_MEMORY_MODES,
    ACTION_MEMORY_NONE,
    DEFAULT_ACTION_CONTEXT_MAX_CHARS,
    DEFAULT_ACTION_MEMORY_MAX_EVENTS,
    _action_system_prompt,
    _build_action_prompt,
    parse_action_diagnostic,
)
from aiworld.llm_client import LLMClient, config_from_env  # noqa: E402
from aiworld.logging_schema import CompletionRecord, TickRecord  # noqa: E402
from aiworld.salience import render_perceived  # noqa: E402
from aiworld.config import load_config  # noqa: E402
from aiworld.self_model import SHARED_DIRECTIVE  # noqa: E402
from aiworld.seed_plan import SeedPlan  # noqa: E402
from aiworld.provenance import (  # noqa: E402
    ACTION_API_VERSION, ANALYSIS_SCHEMA_VERSION, git_revision)
from scripts.run_llm import build_world, config_digest  # noqa: E402


STATIC_OUTCOME_STATUS = "not_applied_static_sampling_null"
STATIC_OUTCOME = "proposal not applied: static-context sampling null"


def _metadata(method: Any) -> dict[str, Any]:
    """Read optional client provenance without requiring a production client in tests."""
    return method() if callable(method) else {}


def _condition_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    agents = cfg.get("agents", {})
    reflection = cfg.get("reflection", {})
    return {
        "show_self_model": agents.get("show_self_model", True),
        # Match run_llm.py's legacy default exactly when this runner is used
        # against an older positive-control config that omitted the field.
        "directive": agents.get("directive", SHARED_DIRECTIVE),
        "action_memory": agents.get("action_memory", ACTION_MEMORY_NONE),
        "action_memory_max_events": agents.get(
            "action_memory_max_events", DEFAULT_ACTION_MEMORY_MAX_EVENTS),
        "action_context_max_chars": agents.get(
            "action_context_max_chars", DEFAULT_ACTION_CONTEXT_MAX_CHARS),
        "reflection_enabled": False,
        "source_reflection_enabled": reflection.get("enabled", True),
        "static_context_sampling_null": True,
    }


def run_static_sampling(
    *,
    world,
    client,
    n_slots: int,
    ticks: int,
    vocabulary: str,
    directive: str | None,
    show_self_model: bool,
    action_memory: str = ACTION_MEMORY_NONE,
    action_memory_max_events: int = DEFAULT_ACTION_MEMORY_MAX_EVENTS,
    action_context_max_chars: int = DEFAULT_ACTION_CONTEXT_MAX_CHARS,
    seed_plan: SeedPlan | None = None,
    on_context_ready: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Log repeated actions from one immutable prompt and return its provenance.

    ``n_slots`` produces labelled repeated samples (``sample_slot_0`` etc.), not
    interacting agents and not independent worlds.  The one template agent and
    World are never resolved or stepped; resource quantities, locations, drives,
    inventory, and memory therefore remain fixed across all calls.

    Every draw gets its OWN sampling seed, derived from ``seed_plan`` (one per
    ``(slot, tick)`` pair). This null's entire point is to measure decoder/model
    variation under one frozen context -- reusing a single seed across every
    draw collapses that variation toward degenerate by construction, which
    would let any nonzero real-condition variation "beat" the null for free
    (docs/EXPERIMENT_PROTOCOL.md evidence gate #5). ``seed_plan`` defaults to
    an arbitrary fixed run_seed only for callers that don't care about
    reproducing a specific draw sequence (e.g. throwaway tests); real runs
    should always pass one built from the run's declared seed.
    """
    seed_plan = seed_plan or SeedPlan(run_seed=0)
    if n_slots < 1:
        raise ValueError("n_slots must be at least 1")
    if ticks < 1:
        raise ValueError("ticks must be at least 1")
    if not world.agents:
        raise ValueError("static sampling null requires one template agent")

    template = world.agents[0]
    perceived = render_perceived(
        world.observations_for(template), template.drives.salience_biases(), top_k=6)
    system_prompt = _action_system_prompt(vocabulary, directive, show_self_model)
    prompt = _build_action_prompt(
        template,
        perceived,
        world,
        show_self_model=show_self_model,
        action_memory=action_memory,
        action_memory_max_events=action_memory_max_events,
        action_context_max_chars=action_context_max_chars,
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    pre_drives = template.drives.snapshot()
    pre_position = template.location
    pre_inventory = world._inventory_snapshot(template)
    opportunities = world._pre_action_opportunities(template)
    provenance = {
        "frozen_context_agent": template.agent_id,
        "frozen_prompt_sha256": prompt_sha256,
        "frozen_prompt_chars": len(prompt),
        "n_sample_slots": n_slots,
        "n_repeats_per_slot": ticks,
        "n_static_action_calls": n_slots * ticks,
        "outcome_status": STATIC_OUTCOME_STATUS,
        "sampling_seed_mode": "distinct_per_call",
    }
    # Metadata must precede the first model call.  This leaves an immutable,
    # interpretable record even if a provider failure interrupts the null run.
    if on_context_ready is not None:
        on_context_ready(dict(provenance))

    for tick in range(1, ticks + 1):
        # LLM completion provenance records the logical sample slot, while the
        # prompt remains exactly the template agent's unchanged context.
        world.tick_count = tick
        for slot in range(n_slots):
            slot_id = f"sample_slot_{slot}"
            call_seed = seed_plan.seed_for("static_sampling", slot=slot, tick=tick)
            raw = client.complete(prompt, system=system_prompt, seed=call_seed)
            action, parse_status = parse_action_diagnostic(raw, world, template)
            proposal_valid = parse_status == "ok"
            world.logger.log_completion(CompletionRecord(
                phase="static_sampling_action",
                tick=tick,
                agent_id=slot_id,
                system_prompt=system_prompt,
                user_prompt=prompt,
                raw_completion=raw,
                parsed_output=action,
                parse_status=parse_status,
                request_metadata=_metadata(getattr(client, "request_metadata", None)),
                response_metadata=_metadata(getattr(client, "response_metadata", None)),
            ))
            world.logger.log_tick(TickRecord(
                run_id=world.logger.run_id,
                tick=tick,
                agent_id=slot_id,
                alive=True,
                # Retain the old required fields, but make it impossible to read
                # these records as post-resolution behavior: all structured state
                # is the single frozen decision context.
                drives=dict(pre_drives),
                position=pre_position,
                chosen_action=action,
                perceived=perceived,
                self_model_version=template.history.current.version,
                proposal_index=slot,
                resolution_index=None,
                proposed_action=action,
                pre_action_drives=dict(pre_drives),
                pre_action_position=pre_position,
                pre_action_inventory=dict(pre_inventory),
                pre_action_opportunities=copy.deepcopy(opportunities),
                parse_status=parse_status,
                proposal_valid=proposal_valid,
                outcome_status=STATIC_OUTCOME_STATUS,
                action_succeeded=None,
                outcome=STATIC_OUTCOME,
                reflection_reason=None,
            ))

    return provenance


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample repeated LLM actions from one frozen initial context.")
    ap.add_argument("--config", default="configs/experiment_factual_memory_unassigned.yaml",
                    help="source condition whose initial context is frozen")
    ap.add_argument("--agents", type=int, default=None,
                    help="number of repeated sample slots; defaults to agents.count")
    ap.add_argument("--ticks", type=int, default=None,
                    help="repeated draws per slot; defaults to run.ticks")
    ap.add_argument("--seed", type=int, default=0,
                    help="world/setup seed; match the paired real-condition seed")
    ap.add_argument("--sampling-seed", type=int, default=None,
                    help="base seed for this null's per-call SeedPlan derivation; "
                         "defaults to AIWORLD_SAMPLING_SEED or --seed. Every draw "
                         "still gets its OWN distinct derived seed -- this flag "
                         "only controls reproducibility of that derivation, never "
                         "a single literal value sent to every call")
    ap.add_argument("--run-id", default="sampling_null",
                    help="immutable output id; reusing one is an error")
    args = ap.parse_args()

    exp_cfg = load_config(args.config)
    n_slots = args.agents if args.agents is not None else exp_cfg["agents"]["count"]
    ticks = args.ticks if args.ticks is not None else exp_cfg["run"]["ticks"]
    vocabulary = exp_cfg.get("world", {}).get("vocabulary", "neutral")
    memory_style = exp_cfg.get("world", {}).get("memory_style", "narrative")
    mortality = bool(exp_cfg.get("world", {}).get("mortality", True))
    condition = _condition_from_config(exp_cfg)
    # Static sampling never reflects, even when the source config is a
    # self-narrative treatment: the null controls action-decoding variation, not
    # identity-generation variation.
    show_self_model = condition["show_self_model"]
    directive = condition["directive"]
    if condition["action_memory"] not in ACTION_MEMORY_MODES:
        choices = ", ".join(sorted(ACTION_MEMORY_MODES))
        raise ValueError(
            f"Unknown agents.action_memory {condition['action_memory']!r}; "
            f"expected one of: {choices}."
        )
    if (condition["action_memory"] == ACTION_MEMORY_FACTUAL_LEDGER
            and memory_style != "factual"):
        raise ValueError(
            "agents.action_memory: factual_ledger requires "
            "world.memory_style: factual."
        )

    llm_cfg = config_from_env()
    base_sampling_seed = (args.sampling_seed if args.sampling_seed is not None
                          else llm_cfg.sampling_seed if llm_cfg.sampling_seed is not None
                          else args.seed)
    # Every call below passes its own explicit per-call seed
    # (seed_plan.seed_for(...)), which always overrides this config default.
    # Leave the config default at None (rather than base_sampling_seed) so
    # completions.jsonl's per-record "sampling_seed" (config snapshot) never
    # disagrees with "effective_sampling_seed" (what was actually sent) --
    # base_sampling_seed only seeds the SeedPlan derivation below, it is
    # never itself sent to the model.
    llm_cfg.sampling_seed = None
    seed_plan = SeedPlan(run_seed=base_sampling_seed)
    client = LLMClient(llm_cfg)
    print(f"Using model '{llm_cfg.model}' at {llm_cfg.base_url} "
          f"(provider={llm_cfg.provider})")

    resolved_cfg = copy.deepcopy(exp_cfg)
    resolved_cfg.setdefault("agents", {})["count"] = n_slots
    resolved_cfg.setdefault("run", {})["ticks"] = ticks
    resolved_cfg["run"]["seed"] = args.seed
    resolved_cfg["run"]["sampling_seed"] = base_sampling_seed
    resolved_cfg["run"]["model"] = llm_cfg.model
    resolved_cfg.setdefault("null", {})["mode"] = "static_context_sampling"
    resolved_cfg["null"]["source_config"] = args.config

    world, setup = build_world(
        resolved_cfg, args.run_id, n_slots, args.seed, mortality=mortality,
        vocabulary=vocabulary, memory_style=memory_style,
        initial_directive=directive)
    # ``build_world`` constructs a full world only to supply the exact initial
    # context that a paired real run would see.  run_static_sampling deliberately
    # leaves that world unchanged thereafter.
    def log_static_metadata(static_provenance: dict[str, Any]) -> None:
        world.logger.log_meta({
            "run_id": args.run_id,
            "policy": "static_sampling_null",
            "seed": args.seed,
            "sampling_seed": base_sampling_seed,
            "mortality": mortality,
            "n_agents": n_slots,
            "condition": condition,
            "resolved_config": resolved_cfg,
            "resolved_config_sha256": config_digest(resolved_cfg),
            "prompts": {
                "action_system": _action_system_prompt(vocabulary, directive, show_self_model),
                "reflection_template": None,
            },
            "llm": client.request_metadata(),
            "action_api_version": ACTION_API_VERSION,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "null_interpretation": (
                "Static-context proposal-only null. It estimates decoder/model variation "
                "under one frozen initial prompt; it is not realized environmental behavior "
                "and must not be pooled with world-run action outcomes. Seed fields: this "
                "record's top-level 'sampling_seed' is only the SeedPlan derivation base -- "
                "it was never sent to the model. Each draw in completions.jsonl carries its "
                "own distinct 'effective_sampling_seed' (what was actually sent); the base "
                "and the per-draw values are expected to differ."
            ),
            **static_provenance,
            **setup,
        })

    status, error = "completed", None
    static_provenance = None
    try:
        static_provenance = run_static_sampling(
            world=world,
            client=client,
            n_slots=n_slots,
            ticks=ticks,
            vocabulary=vocabulary,
            directive=directive,
            show_self_model=show_self_model,
            action_memory=condition["action_memory"],
            action_memory_max_events=condition["action_memory_max_events"],
            action_context_max_chars=condition["action_context_max_chars"],
            seed_plan=seed_plan,
            on_context_ready=log_static_metadata,
        )
    except Exception as exc:
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
    print(f"Static sampling null '{args.run_id}' done. "
          f"{static_provenance['n_static_action_calls']} frozen-context draws. "
          f"LLM stats: {client.stats()}")
    print(f"Logs in runs/{args.run_id}/")


if __name__ == "__main__":
    main()
