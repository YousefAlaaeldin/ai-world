# AI World

An open-ended, multi-agent simulation experiment. Agents begin **identical**, with a
single shared directive — *"survive and find your purpose"* — no assigned identity, no
assigned goal. Their character is a mutable document they author and revise from
experience. The research question is not "do agents follow their role?" but **"what do
agents become when nothing is assigned?"**

Originally scoped as a [DeepMind Concordia](https://github.com/google-deepmind/concordia)
extension; the design settled instead on a small custom tick-loop engine
(`aiworld/sim/world.py`) that resolves actions itself (a Game-Master equivalent) and
runs against a **local model via [Ollama](https://ollama.com)** — no hosted API
dependency, full control over the mechanics, and testable end-to-end with zero model
calls. The novel contribution is a `SelfModel` component (identity as a state
variable the agent writes to) and a `Drive` system (hunger, thirst, affiliation) that
biases *attention*, not behavior — so purpose stays genuinely open.

## The central hypothesis

Identical initial conditions → divergent trajectories. If self-models converge to a
monoculture, the "freedom" is illusory (environment or prompt is deterministic). If they
diverge, **the divergence pattern is the result** — and recurring purpose-clusters across
independent runs would be the headline finding.

## Status

Phase 0 complete (mechanics + logging, 12/12 unit tests passing). Phase 1 in
progress: the tick loop, action resolution, and reflection step all run end-to-end
against the stub (no-LLM) policy; a live run against a local model is next. See
`docs/DESIGN.md` for the full experimental design and `docs/ROADMAP.md` for build
order.

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

Default backend is a **local model via Ollama** — no API key needed. Pull the
default model (`ollama pull huihui_ai/mistral-small-abliterated:24b`) and it's used
automatically; abliterated so it isn't steered by refusal/safety-tuning artifacts
when deciding what to do, which matters here since we don't want the model's action
choices contaminated by anything beyond the shared directive. To use a different
model or a hosted provider instead, copy `.env.example` to `.env`, fill in the
relevant key, and set `AIWORLD_PROVIDER` / `AIWORLD_BASE_URL` / `AIWORLD_MODEL`
(see `aiworld/llm_client.py:config_from_env`).

## Running

```bash
# Mechanics smoke test — no LLM, zero cost:
python scripts/run_stub.py --run-id smoke01

# Real run against the local model, reading configs/experiment_minimal.yaml:
python scripts/run_llm.py --run-id live01

# Any config value can be overridden per-run, e.g.:
python scripts/run_llm.py --run-id live02 --agents 8 --ticks 60 --prompt leading

# A single run is one data point (DESIGN.md §6: "n=1 means nothing with
# stochastic agents"). Sweep every seed in a config's run.seeds list instead:
python scripts/run_sweep.py --config configs/experiment_minimal.yaml --policy llm

# The environment-contrast pair (DESIGN.md §6/§9) — same seeds, environment
# stripped of every differentiating signal, for "does environment matter at all":
python scripts/run_sweep.py --config configs/experiment_null_baseline.yaml --policy llm
```

Logs land in `runs/<run_id>/` (`self_models.jsonl`, `ticks.jsonl`, `events.jsonl`,
`meta.json`) — git-ignored, since they're experiment output, not source. A sweep
writes one such directory per seed under `runs/<experiment_name>/seed_<k>/`.

**Start small**: 5–8 agents, one resource, mortality on, ~20 repeat runs from the
identical seed-state before adding any complexity.
