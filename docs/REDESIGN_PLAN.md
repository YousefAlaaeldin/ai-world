# Redesign Plan — de-primed baseline + behavioral measurement

**Paper claim:** *"Emergent behavioral organization under unassigned objectives in
LLM agents."* Note what this commits the work to: the outcome is **behavioral
organization** (structure in what agents *do*), and "unassigned" is the load-bearing
adjective — so the primary result must come from the de-primed baseline measured by
behavior, not from self-report text under a "survive/find your purpose" directive.
Tranche 3's metrics are what substantiate "behavioral organization"; Tranche 1's
de-priming is what earns "unassigned."

Driven by external review (see conversation 2026-09-14). The project's defensible
question is **not** "agents with no priors" (no LLM is a blank slate) but:

> **No experimenter-specified identity, value hierarchy, or distal objective —
> observe whether stable *behavioral* commitments emerge from interaction with an
> open environment.**

The shift: stop telling agents to "survive and find your purpose," stop showing them
a self-model and feeding it back into decisions, record **behavior** first, and infer
candidate purposes post-hoc from behavior that is stable, costly, predictive, and
robust — not from a self-report text field. Self-modeling becomes a *treatment*, not
the baseline.

## Conditions matrix (target)

| Condition | World | Self-model fed back? | Directive? | Tests |
|---|---|---|---|---|
| Sampling null | static/yoked | no | no | divergence from sampling alone |
| **Open-world baseline** | rich | **no** | **no** | behavioral organization without identity priming |
| Factual-memory arm | rich | no (factual ledger only) | no | whether memory enables stable strategy |
| Self-narrative arm | rich | **yes** | no | whether self-authored identity amplifies/stabilizes |
| Original-prompt arm | rich | yes + purpose directive | yes | positive control for priming (today's setup) |

**The load-bearing comparison:** open-world baseline vs. self-narrative arm. If behavior
only stabilizes once "Who I am / What matters" is fed back, that is a result about
*self-conditioning*, not unassigned purpose.

## "Emergent purpose" defined behaviorally

A pattern is a candidate emergent purpose only if it is: (1) not instructed, (2) stable
over time, (3) costly/selective (persists when an easier equally-viable alternative
exists), (4) predictive of choices in new situations, (5) robust to relabeling /
paraphrase / held-out maps, (6) observable in behavior, not only self-description.

---

## Tranches (each independently runnable + testable)

### Tranche 1 — condition scaffolding + fix the vocabulary leak  ← IN PROGRESS
- `primed` / `show_self_model` / `directive` switches so an arm can omit the self-model
  block, the "survive/find your purpose" directive, and all "who you are" framing from
  the live prompt; reflection becomes optional.
- **Fix review issue #2:** route the outcome / memory / crisis strings
  (`world.py` `consumed {drive} resource`, `... ran out`, `nearly died from {drive}`)
  through the vocabulary layer, so the neutral arm no longer leaks `hunger`/`thirst`
  into the *reflection* prompt via `recent_memories`. Extend the vocab test to the
  reflection path (the action-prompt-only test was the exact hole the review found).
- Defaults preserve today's behavior (self-narrative + directive = the *original-prompt
  arm* / positive control). No new runnable baseline config yet — see the open tradeoff.

### Tranche 2 — factual memory + opaque vocabulary arm
- Replace narratively-loaded memory ("nearly died", "ran out", "salient") with a factual
  ledger: `tick 15: used R3; S1 0.31 -> 0.77`, `tick 16: message from A7`.
- Add an `opaque` vocabulary arm (`S1/S2`, `R1/R2/R3`) alongside neutral/biological.

### Tranche 3 — behavioral metrics (unblocks the baseline)
- Primary outcome becomes a **behavioral trajectory**, not self-model embedding distance:
  resource acquisition/use/storage/sharing, exploration coverage, discretionary-vs-
  viability time, social-network formation, public-good investment, strategy persistence
  after perturbation, held-out trade-off probes.
- Cluster behavior on a **fixed representation trained once across all conditions**;
  choose k by stability/null tests, not a forced k=5.
- End-only / held-out self-report as *secondary* evidence, never fed back into actions.

### Tranche 4 — richer action space (multiple viable lives)
- gather/store, give/withhold, build/maintain shared infrastructure, explore, trade,
  form/break agreements, generated messages, public-vs-private investment.
- Make several strategies similarly viable — if one behavior is necessary to avoid
  death, it is adaptation, not emergent purpose.

### Tranche 5 — remove hidden asymmetries
- Simultaneous action proposals + fair conflict resolution; randomize/counterbalance
  execution order each tick (fixes fixed-ID first-mover advantage).
- Randomize opaque location/resource labels across seeds; balance starts/maps.
- No automatic self-model transmission on `talk` (add generated-dialogue or
  no-communication baseline).
- Condition-specific, immutable run folders (fixes the sweep-append contamination,
  review #3); log model version, full resolved config, prompts, seed, decoding params,
  action order, raw completions, and parse failures (review reproducibility gap).
- Send an LLM sampling `seed` (currently only temperature is set — not reproducible).
- Guard reflection so a just-died agent doesn't reflect; separate crisis-triggered
  reflections from scheduled ones so mortal/scarce arms don't get extra identity-
  generation "dosage" (review #4).

---

## Open tradeoff to resolve (needs the user)

With reflection off, `SelfModelHistory` never advances past v0, so `divergence_over_time`
and `cluster_final_purposes` return **nothing** for the baseline — the baseline has no
analyzable outcome until Tranche 3 lands. Options: (a) land Tranche 1 as pure
capability + leak-fix and don't run a baseline experiment until Tranche 3; or (b) pull a
minimal behavioral feature extractor forward into Tranche 1. Chosen: **(a)** — Tranche 1
ships scaffolding + the bug fix with today's behavior preserved as the default arm; no
un-analyzable baseline config is shipped.

**Resolved:** the paper's central claim is *"Emergent behavioral organization under
unassigned objectives in LLM agents"* (recorded at the top of this file). This pins
Tranche 3's primary outcome to **behavioral** organization measured in the de-primed
(unassigned) baseline — self-report is secondary and never fed back.

**Still pending:** propagate this reframing (behavioral-organization claim; de-primed
baseline as the primary condition) into the three paper docs — PAPER_OUTLINE,
RESEARCH_PROPOSAL, RESEARCH_PITCH — which still carry the older "what do agents
become when nothing is assigned" framing.
