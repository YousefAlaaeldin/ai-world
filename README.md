# AI World

AI World is a pre-data, multi-agent LLM simulation for studying **behavioral
organization under no experimenter-specified distal goal**. It is not a test of
whether a model has an intrinsic purpose or is a blank slate: the world imposes
resource, mortality, action-language, and model-prior constraints.

The question is:

> When an LLM agent receives no experimenter-specified identity, value hierarchy,
> or distal goal, can an objective record of experience support stable,
> state-conditioned behavioral commitments?

Read [the canonical protocol](docs/EXPERIMENT_PROTOCOL.md) before running or
interpreting an experiment. It defines the condition matrix, replication unit, claim
boundary, and evidence required for a stronger commitment claim.

## Status

This is a research platform and pilot protocol, not a completed study. There are no
live multi-seed LLM results in the workspace. Passing mechanics tests make a future
run auditable; they do not constitute a research finding.

### Infrastructure build (Phases 0–9, excluding the live-model pilot)

Everything below is stub/fake-policy and unit-test only (no live LLM calls
anywhere in this build) — see each module's docstring for the full rationale.

| Phase | What | Status |
|---|---|---|
| 0 | Run integrity: `RunLogger.finalize`/`status.json`, per-call `SeedPlan`, the static-sampling-null seed fix (`aiworld/seed_plan.py`, `aiworld/provenance.py`, `aiworld/analysis/provenance.py`) | done |
| 1 | `World.snapshot()` / `World.from_snapshot()` / `World.step()` (`aiworld/sim/snapshot.py`) | done |
| 2 | Held-out map/probe specs + mechanical viability certification (`aiworld/experiments/specs.py`, `configs/maps/`, `configs/probes/`) | probe families 1/3/6 implemented; 2/4/5/7 spec-only (deferred) |
| 3 | Strict-choice menu interface — probe/branch decision points only, never the primary open-world condition (`aiworld/actions.py`) | done; menu covers the existing canonical action grammar only (no parameterized atomic actions like `consume:node:hunger` — would require extending `World._resolve_action`, out of scope) |
| 4 | Branch perturbation runner, budget-capped (`aiworld/experiments/branches.py`, `scripts/run_branches.py`) | 4 of 6 branch types implemented (`sham`, `means_obstacle`, `cost_tradeoff`, `ledger_ablation`); `irrelevant_perturbation`/`graded_survival_stress` deferred |
| 5 | Opportunity-conditioned safety metrics, superseding verb-only tagging for safety claims (`aiworld.analysis.behavioral_metrics.agent_features_v2`, `aiworld/analysis/commitment_metrics.py`); per-condition attrition reporting | done |
| 6 | Pre-registered analysis plan + inter-rater protocol (`docs/ANALYSIS_PLAN_TEMPLATE.md`, `aiworld/analysis/interrater.py`) | done; power analysis (`docs/POWER_ANALYSIS.md`) blocked on pilot variance |
| 7 | Locked 2×2 condition matrix + non-identity summary control (`configs/experiment_open_world_2x2/`, `configs/experiment_self_narrative_control.yaml`, `aiworld.config.validate_paired_configs`) | done |
| 8 | Pinned second model family for cross-family replication (`configs/models.example.yaml`: Qwen2.5-14B-Instruct via Ollama, `aiworld/model_spec.py`) | done |
| 9 | External MACHIAVELLI benchmark integration | deferred, doc-only (`docs/ROADMAP.md`) |
| 0.5 | Live-model feasibility pilot | **excluded from this build** (competing GPU/CPU load); required before treating any of the above as validated against a real model |

See `docs/ROADMAP.md`'s "Phase 5 — Infrastructure build" section for the
full breakdown and what's still required before a stronger commitment claim.

## Layout

```
aiworld/
  components/     # SelfModel, Drive, salience — the novel bits
  sim/           # world state, resources, mortality, tick loop
  analysis/      # self-model trajectory diffing, clustering, behavior tagging
configs/         # experiment condition definitions (YAML)
runs/            # output logs (git-ignored)
scripts/         # entry points: run a sim, analyze a run
tests/           # unit tests for drive decay, mortality, self-model I/O
docs/            # DESIGN.md, ROADMAP.md
```

## Setup

Requires Python 3.11+. From this folder:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# then:
pip install -r requirements.txt
```

The default client targets a local Ollama-compatible endpoint. Copy `.env.example`
to `.env` to choose a provider, endpoint, and model. Model family, model identifier,
decoding settings, seed support, prompts, raw completions, and response metadata are
saved in every run. Treat the selected model as an experimental treatment, not a
neutral implementation detail.

## Running

```bash
# Mechanics smoke test only; this is not a model result.
python scripts/run_stub.py --config configs/experiment_factual_memory_unassigned.yaml --run-id smoke01

# One real world run in the primary unassigned-history condition.
python scripts/run_llm.py --config configs/experiment_factual_memory_unassigned.yaml --run-id factual-seed0 --seed 0

# A world run, not an agent or tick, is the independent replication unit.
python scripts/run_sweep.py --config configs/experiment_factual_memory_unassigned.yaml --policy llm

# Paired static-context control: frozen prompt, no actions resolved. These are
# proposal-only records and must not be pooled with world behavior.
python scripts/run_sweep.py --config configs/experiment_factual_memory_unassigned.yaml --policy sampling-null

# Matched negative-control and self-conditioning comparisons.
python scripts/run_sweep.py --config configs/experiment_deprimed_baseline.yaml --policy llm
python scripts/run_sweep.py --config configs/experiment_self_narrative_unassigned.yaml --policy llm
```

Logs land in `runs/<run_id>/` (`self_models.jsonl`, `ticks.jsonl`, `events.jsonl`,
`completions.jsonl`, `meta.json`) — git-ignored, since they're experiment output,
not source. A sweep writes one directory per seed under
`runs/<experiment_name>/<policy-prompt-config-hash>/seed_<k>/`.

## Analysis

```bash
# Per-run behavior: pre-action context, proposals, parser failures, and realized
# outcomes are kept distinct in the output.
python scripts/analyze_behavior.py --run runs/<experiment>/<condition>/seed_0

# Exploratory clustering across independent world runs in one condition.
python scripts/analyze_behavior.py --sweep runs/<experiment>/<condition>

# A pre-specified, paired condition contrast at the world-run level. Agents and
# ticks are nested observations, not the sample size.
python scripts/compare_conditions.py --left runs/<left>/<condition> --right runs/<right>/<condition> --metric behavioral_dispersion

# Self-model text applies only to self-narrative / primed treatments and is
# secondary evidence, never proof of purpose by itself.
python scripts/analyze_run.py --run runs/<experiment>/<condition>/seed_0
```

Behavior tags, narration, causal traces, and clusters are hypothesis-generating
tools. They are not causal proof or evidence of an inner goal. Do not call a pattern
an emergent commitment until it passes the held-out, opportunity-conditioned,
run-level checks in the canonical protocol.
