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
