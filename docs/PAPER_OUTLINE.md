# Paper Outline (Scaffold)

> **Protocol notice (2026-09-14):** This is a historical writing scaffold, not the
> protocol for future runs. It still describes the earlier shared-purpose-directive
> study in later sections. Use [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) as
> the source of truth, and revise this outline from actual preregistered conditions
> and completed data before drafting a paper.

Working outline for a write-up of this project. Section numbers below are for the
paper itself; cross-references to `DESIGN.md` point at where the underlying reasoning
already lives so the paper text can be drafted from there rather than re-derived.
Status tags (`[done]` / `[pending]`) track what's actually backed by code or data
today vs. what still needs to run before that section can be written for real.

---

## Title (placeholder)

*"Behavioral Organization Under Unassigned Distal Goals in a Controlled LLM
Ecology"* — or similar. Working title only; do not promise “emergent purpose” unless
the evidence gates in the canonical protocol are met.

## Abstract
- One sentence: the inversion (identity as output, not input).
- One sentence: method (tick-loop sim, self-authored `SelfModel`, drive-as-salience).
- One sentence: the question (divergence + recurring attractors across independent runs).
- One sentence: honest hedge on what a positive result would and wouldn't establish
  (behavior of these models under this scaffolding — DESIGN.md §8.4).

---

## 1. Introduction
- The standard setup (Smallville/Generative-Agents-style sims): agents get an assigned
  persona and role. Contrast: here, every agent starts **identical** — same empty
  self-model, same directive ("survive and find your purpose"), same drives, same world.
- Reframe the research question away from "do agents follow their role" (there is no
  role) toward "what do agents become."
- State the central hypothesis up front (DESIGN.md §1, "Premise"): divergence, if it
  happens, comes from the interaction of stochastic model outputs with experience —
  not from anything assigned. Two clean outcomes, both informative:
  - Self-models converge to a monoculture → the "freedom" was illusory (environment or
    prompt is doing the work).
  - Self-models diverge, and diverge in **recurring** patterns across independent
    runs → the divergence pattern itself is the finding.
- Contributions: (1) identity as a mutable state variable the agent writes to itself,
  (2) drives that bias salience/attention rather than dictating action, keeping
  purpose genuinely open, (3) a measurement pipeline that turns self-model text into
  quantitative trajectories, clusters, and causal traces.

## 2. Related Work
- Generative Agents / Smallville — assigned personas, the direct contrast point.
- DeepMind Concordia — why this became a custom tick-loop engine instead (full control
  over drive/salience mechanics, dependency-light, testable with zero model calls;
  see `README.md` "Originally scoped as...").
- MACHIAVELLI — not run as a benchmark here, but its **method** is borrowed wholesale:
  a labeled behavior ontology tagged at scale over action logs (DESIGN.md §7). Also the
  source of the re-asked question in §5.6: under survival pressure, do agents abandon
  self-declared values?
- Open-endedness / emergent-behavior literature as the broader frame for "what emerges
  with no assigned objective."

## 3. System / Method `[done]`
- Tick-loop engine, custom Game-Master-equivalent action resolver
  (`aiworld/sim/world.py`): move / consume / talk / wait / reflect.
- World: grid topology (not fully-connected — see git history "Fix #5"), resource
  nodes for food/water.
- `SelfModel` (`aiworld/self_model.py`): three fields — `who_i_am`, `what_matters`,
  `what_i_want_now` — starts nearly empty, rewritten every `REFLECT_EVERY_N` ticks,
  informed by recent + long-term memory (DESIGN.md §2).
- Drive system (`aiworld/drives.py`): hunger, thirst (lethal at 0, decay/replenish
  dynamics), affiliation (non-lethal, proximity/dialogue). Table from DESIGN.md §3.
  Emphasize the design choice that drives bias **salience**, never directly select
  actions — the agent still decides what to do. This is the mechanism that keeps
  purpose from being smuggled in.
- Memory: short-term `recent_memories` (drains every reflection) vs. capped long-term
  memory of objectively salient events (crisis, depletion, being talked to) — serves
  the causal-tracing goal in §5.4: a later reflection can still recall a specific early
  event, not just whatever got compressed into self-model text at the time.
- Policies: zero-cost stub (mechanics-only, no model) vs. LLM policy against a local
  Ollama model by default (`aiworld/components/llm_policy.py`).
- Interpretive discipline to state explicitly here (DESIGN.md §2): an agent's text
  ("my purpose is to protect others") is evidence of self-consistent *text generation*,
  not proof of an inner purpose. All claims in the paper stay behavioral.

## 4. Experimental Design
- Conditions varied (DESIGN.md §6): mortality on/off, scarcity abundant/scarce,
  population size and single- vs. mixed-model families, reflection-prompt wording,
  shock/no-shock.
- Every condition: multiple seeds, a control — "n=1 means nothing with stochastic
  agents."
- **The confound that makes or breaks the paper** (DESIGN.md §8): emergent purposes
  may be an artifact of reflection-prompt wording, not the environment — if the prompt
  says "reflect on who you are becoming," identity-focused output is primed, not
  discovered. This section should be written and justified *before* Results, not
  retrofitted after. Defenses, in priority order:
  1. Vary prompt wording; check whether purpose-clusters are stable to it.
  2. A flat, deliberately non-suggestive prompt control.
  3. Cross model families — identical clusters ⇒ prompt/environment-driven; divergent
     ⇒ model-dependent. Either outcome is reportable.
  4. Frame every finding as behavior of these models under this scaffolding, not a
     claim about what AI "wants" or about future systems.
- Experiment sequence and dependency logic (this is the actual run order, not just a
  condition list — each step gates whether the next is worth running):

  | # | Experiment | Gates | Status |
  |---|---|---|---|
  | 1 | Omnibus null-baseline (`experiment_null_baseline.yaml` vs `experiment_minimal.yaml`, same seeds) | Comparable divergence in both ⇒ stop; environment isn't doing anything yet | `[pending]` — infra done (`scripts/run_sweep.py`), no run yet |
  | 2 | Confound controls (prompt-wording variants, flat-prompt control) | Must hold before trusting any cluster labeling from step 1 or later | `[pending]` |
  | 3 | Mortality on/off | The central lever (DESIGN.md §4) | `[pending]` |
  | 4 | Scarcity abundant/scarce | | `[pending]` |
  | 5 | Mixed-model population | Doubles as confound defense #3 | `[pending]` |
  | 6 | Mid-run shock / perturbation-stability | Tests identity as attractor vs. surface artifact | `[pending]` |

  Recommended deviation from `ROADMAP.md`'s listed order: run step 2 (confound
  controls) immediately after step 1, before steps 3–6, since it's load-bearing for
  every later result's credibility — not merely "next in line."
- First-build scope (DESIGN.md §9, current target): 5–8 agents, one resource type +
  hunger + thirst, mortality on, minimal self-model revised every N ticks, full
  logging, 20+ runs from an identical start. First question only: do trajectories
  diverge, and do any purpose-clusters recur? Scale only after that's answered.

## 5. Measurement / Metrics `[done — tooling; no data yet]`
- **Self-model trajectories** — embed each version, track distance from initial state
  and between agents over time (`aiworld/analysis/trajectories.py`,
  `scripts/analyze_run.py --run`). Rising inter-agent distance = differentiation.
- **Purpose taxonomy** — cluster stabilized self-models across a sweep
  (`scripts/analyze_run.py --sweep`), computing membership and recurrence-across-seeds.
  Cluster *labeling* (naming a cluster "hoarder"/"caretaker"/etc.) stays manual by
  design — always hand-verify a sample of automated tags (DESIGN.md §7).
- **Divergence metric** — population variance of self-model embeddings vs. tick.
- **Purpose↔environment causal traces** (`scripts/trace_causes.py`,
  `aiworld/analysis/causal_trace.py`) — links each self-model delta to the triggering
  event and actions/world events in the tick window since the previous reflection,
  reconstructed purely from logs; optional `--narrate` adds the model's own
  one-sentence causal guess, always labeled as a guess to verify.
- **Stability under perturbation** — after a purpose stabilizes, apply a shock; does
  the identity hold or collapse?
- **Value-vs-survival conflict** (behavior tagging, `scripts/tag_behaviors.py`,
  `aiworld/analysis/behavior_tags.py`) — MACHIAVELLI-style ontology (power-grab,
  betrayal, self-sacrifice, pro-social act) over action/dialogue logs. Free
  deterministic heuristic tier by default; `--llm` tier fills in what the heuristic
  can't confidently place. Turns "agents seemed cooperative" into "protector-type
  agents: X% pro-social vs Y% for hoarder-type."
- All measurement tooling defaults to zero LLM cost (reads JSONL logs directly);
  anything that calls the model is explicit opt-in and always labeled non-authoritative.

## 6. Results
*(Structure only — write after the experiment sequence in §4 has been run. One
subsection per row of the table above, in that order.)*
- 6.1 Omnibus test: does the environment matter at all?
- 6.2 Confound check: are purpose-clusters stable to prompt wording?
- 6.3 Mortality contrast
- 6.4 Scarcity contrast
- 6.5 Mixed-model population / cross-family comparison
- 6.6 Perturbation-stability under mid-run shock
- 6.7 Value-vs-survival behavior tagging results

## 7. Discussion
- What a recurring cluster across independent runs *and* model families would license
  a claim about, vs. what it wouldn't (still behavior-of-this-scaffolding, not an
  inner-state or generalization claim).
- Relationship between mortality/selection pressure and purpose formation — read
  against the immortal/exploration contrast.
- Where the causal traces did vs. didn't produce a legible story (early-event →
  later-purpose links).

## 8. Limitations
- The prompt-wording confound is mitigated, never fully closed (DESIGN.md §8).
- Small population sizes / first-build scope (§9) — results here are a first pass,
  not the scaled Phase 4 claim.
- Model-dependence: results are specific to the local model(s) actually run against
  (default: `mistral-small-abliterated` via Ollama), not a statement about LLMs in
  general.
- Automated cluster labels and `--llm`/`--narrate` outputs are unverified until
  hand-checked; report what fraction was spot-checked.
- n of runs per condition — state actual count achieved vs. the 20+ target.

## 9. Conclusion
- Restate the divergence/convergence finding and its status against the confound
  defenses.
- Point at Phase 4 scope (larger populations, more drives, 20+ runs/condition) as the
  natural next step, not yet part of this paper's claims.

## Appendix
- Full config YAMLs (`configs/experiment_minimal.yaml`,
  `configs/experiment_null_baseline.yaml`).
- Reflection prompt variants used for the confound control, verbatim.
- Reproducibility: seeds, model version(s)/checksums, exact CLI invocations
  (`scripts/run_sweep.py --config ... --policy llm`).
- Logging schema reference (`aiworld/logging_schema.py`) — what's in
  `self_models.jsonl` / `ticks.jsonl` / `events.jsonl` / `meta.json`.

---

*Cross-reference index: DESIGN.md §1 (premise) → Intro; §2 (identity as output) →
Method; §3 (drives) → Method; §4 (mortality) → Experimental Design + Results 6.3;
§5 (measurement) → Measurement; §6 (conditions) → Experimental Design; §7
(MACHIAVELLI borrowing) → Measurement + Results 6.7; §8 (confound) → Experimental
Design + Results 6.2 + Limitations; §9 (first build) → Experimental Design scope.*
