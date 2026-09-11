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
- [ ] Live run against the local model (`python scripts/run_llm.py`). Confirm it
      *runs* end-to-end and produces sane self-model revisions before trusting output.

## Phase 2 — Measurement
- [ ] Embed self-model versions, compute trajectory + divergence metrics
- [ ] Cluster stabilized self-models, auto-label clusters
- [ ] LLM behavior-tagger over action logs (MACHIAVELLI-style ontology)
- [ ] Causal trace tool: link self-model deltas to triggering events

## Phase 3 — Experiments
- [ ] Run the confound controls FIRST (vary reflection prompt, flat control)
- [ ] Mortality on/off contrast
- [ ] Scarcity contrast
- [ ] Mixed-model population
- [ ] Mid-run shock / perturbation-stability

## Phase 4 — Scale & write-up
- [ ] More drives (affiliation), larger population
- [ ] Aggregate across 20+ runs per condition, report distributions not anecdotes
- [ ] Writeup with the honest-limits framing from DESIGN.md §8

## Cost discipline
LLM calls = agents × ticks × (act + reflect + annotate). This explodes fast. Keep
populations small, cache aggressively, batch per tick, and prototype logic in the
pure-Python layer (no model) wherever possible.
