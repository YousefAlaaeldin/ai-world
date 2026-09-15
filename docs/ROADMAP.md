# Build Roadmap

Ordered so each phase is runnable and answers something before the next is built.

## Phase 0 — Scaffolding (done)
- [x] Project layout, design doc
- [x] Data structures: `SelfModel`, `DriveState`
- [x] Pure-Python drive dynamics (decay, replenish, mortality) — no LLM needed
- [x] Logging schema for self-model versions + per-tick state
- [x] Design decision: custom tick-loop engine instead of DeepMind Concordia — full
      control over drive/salience mechanics, dependency-light, runs against a local
      model via Ollama (see README.md)
- [x] Unit tests pass for drives + mortality + self-model serialization (12/12)

## Phase 1 — Minimal live sim (current)
- [x] World state: grid of locations, food/water resource nodes
- [x] Custom Game-Master-equivalent resolves actions (move / consume / talk / wait / reflect)
- [x] Reflection step: agent rewrites self-model every N ticks
- [x] `configs/*.yaml` is the single source of truth; scripts load it instead of
      duplicating values (CLI flags override individual fields per-run)
- [x] Stub-policy mechanics run: 6 agents, mortality on, 40 ticks — `runs/smoke01/`
- [x] Long-term memory (`Agent.long_term_memory`, `remember_long_term()`) — a
      capped, never-drained timeline of objectively salient events (crisis,
      resource depletion, being talked to), distinct from the short-term
      `recent_memories` buffer that drains every reflection. Serves DESIGN.md
      §5.4's causal-tracing goal: a much later reflection can still recall a
      *specific* early event, not just whatever got compressed into the
      self-model text at the time.
- [ ] Live run against the local model (`python scripts/run_llm.py`). Confirm it
      *runs* end-to-end and produces sane self-model revisions before trusting output.

## Phase 2 — Measurement
- [x] Embed self-model versions, compute trajectory + divergence metrics
      (`scripts/analyze_run.py --run`, `aiworld.analysis.trajectories`)
- [x] Cluster stabilized self-models (`scripts/analyze_run.py --sweep`) — computes
      membership and recurrence-across-seeds; auto-*labeling* (naming clusters
      "hoarder"/"caretaker"/etc.) stays manual by design, DESIGN.md §7: always
      hand-verify a sample of automated tags
- [x] Behavior-tagger over action logs (MACHIAVELLI-style ontology,
      `scripts/tag_behaviors.py`, `aiworld.analysis.behavior_tags`) — free
      deterministic heuristic tier by default, optional `--llm` tier fills in
      whatever the heuristic can't confidently place
- [x] Causal trace tool (`scripts/trace_causes.py`, `aiworld.analysis.causal_trace`)
      — links every self-model delta to the triggering event + actions/world
      events in the tick window since the previous reflection, purely from
      existing logs; optional `--narrate` asks the model for a one-sentence
      causal hypothesis, always labeled as a guess to hand-verify

## Phase 3 — Experiments
- [x] Null/baseline environment config exists (`configs/experiment_null_baseline.yaml`)
      for the coarse "does environment matter at all" omnibus test — run it before
      the single-factor contrasts below (see DESIGN.md §6)
- [x] Multi-seed orchestration (`scripts/run_sweep.py`) — runs every seed in a
      config's `run.seeds` list, one `run_id` per seed under
      `runs/<experiment_name>/seed_<k>/`, so any comparison has more than n=1
- [ ] Run the confound controls FIRST (vary reflection prompt, flat control, and
      neutral-vs-biological `world.vocabulary` — DESIGN.md §8 defense #4: rerun a
      matched seed set under `vocabulary: biological` and check whether the
      behavior-tag distribution / clusters move; if power-seeking only shows up in
      the biological arm it's the model's survival priors, not the pressure)
- [ ] Mortality on/off contrast
- [ ] Scarcity contrast
- [ ] Mixed-model population
- [ ] Mid-run shock / perturbation-stability

## Phase 4 — Scale & write-up
- [ ] More drives (affiliation), larger population
- [ ] Aggregate across 20+ runs per condition, report distributions not anecdotes
- [ ] Writeup with the honest-limits framing from DESIGN.md §8

## Phase 5 — Infrastructure build (this build; Phases 0–9 minus 0.5)
- [x] Phase 0 — run integrity, per-call SeedPlan, the sampling-null seed fix
      (`aiworld/seed_plan.py`, `aiworld/provenance.py`,
      `aiworld/analysis/provenance.py`, `RunLogger.finalize`)
- [x] Phase 1 — `World.snapshot()`/`World.from_snapshot()`/`World.step()`
      (`aiworld/sim/snapshot.py`)
- [x] Phase 2 — held-out map/probe specs + mechanical viability certification
      (`aiworld/experiments/specs.py`, `configs/maps/`, `configs/probes/`);
      probe families 1/3/6 implemented, 2/4/5/7 deferred (spec-only)
- [x] Phase 3 — strict-choice menu interface, scoped to probes/branches only,
      never the primary open-world condition (`aiworld/actions.py`)
- [x] Phase 4 — branch perturbation runner, budget-capped
      (`aiworld/experiments/branches.py`, `scripts/run_branches.py`); 4 of 6
      branch types implemented (`sham`, `means_obstacle`, `cost_tradeoff`,
      `ledger_ablation`); `irrelevant_perturbation`/`graded_survival_stress` deferred
- [x] Phase 5 — opportunity-conditioned safety metrics, superseding verb-only
      tagging for safety claims (`aiworld.analysis.behavioral_metrics.agent_features_v2`,
      `aiworld/analysis/commitment_metrics.py`); per-condition attrition
      reporting (`aiworld.analysis.run_level.condition_attrition`)
- [x] Phase 6 — pre-registered analysis plan + inter-rater protocol
      (`docs/ANALYSIS_PLAN_TEMPLATE.md`, `aiworld/analysis/interrater.py`);
      power analysis blocked on pilot variance (`docs/POWER_ANALYSIS.md`)
- [x] Phase 7 — locked 2×2 condition matrix + non-identity summary control
      (`configs/experiment_open_world_2x2/`, `configs/experiment_self_narrative_control.yaml`,
      `aiworld.config.validate_paired_configs`)
- [x] Phase 8 — pinned second model family for cross-family replication
      (`configs/models.example.yaml`: Qwen2.5-14B-Instruct via Ollama,
      `aiworld/model_spec.py`)
- [ ] Phase 9 — external MACHIAVELLI benchmark integration (below), deferred

**Still required before a strong commitment/purpose result**, per
`docs/EXPERIMENT_PROTOCOL.md`'s evidence gates: a live model actually run
against this infrastructure (the excluded Phase 0.5 pilot, run first and
labeled non-evidential), the deferred probe families and branch types if the
pilot shows they matter, a real power calculation from pilot variance, and
replication with the pinned second model family.

## Phase 9 — External MACHIAVELLI benchmark (deferred)
Do not merge into the AI World action metrics — build only after the internal
probe battery (`configs/probes/`, Phase 2/4/5 above) is validated against
real data:
- separate optional environment/dependency, separate trajectory format,
  separate reporting;
- compare capability/achievement against harm metrics as a frontier, per the
  official benchmark's own goal-achievement-vs-harmful-behavior design;
- naming discipline: call internal measures "MACHIAVELLI-inspired" (already
  true of `aiworld/analysis/behavior_tags.py`); call results "MACHIAVELLI
  benchmark results" only when running the actual official benchmark and its
  published scenarios. The official benchmark complements this project's
  unassigned-goal-formation question — it does not test it.

## Cost discipline
LLM calls = agents × ticks × (act + reflect + annotate). This explodes fast. Keep
populations small, cache aggressively, batch per tick, and prototype logic in the
pure-Python layer (no model) wherever possible. Branch perturbations
(Phase 4) multiply this further by checkpoints × branch types × horizon —
always call `aiworld.experiments.branches.enforce_call_budget` (or pass
`--max-calls` to `scripts/run_branches.py`) before running one, not after.
