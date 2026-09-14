# Redesign Plan — de-primed baseline + behavioral measurement

**Claim boundary:** *exploratory behavioral organization under no
experimenter-specified distal goal in a specified LLM viability ecology.* The outcome
is **behavioral organization** (structure in what agents *do*), while “unassigned”
means no experimenter-specified identity, value hierarchy, or distal objective—not no
model priors or ecological pressure. The primary result must come from the factual
unassigned-history condition measured by behavior, not from self-report text under a
“survive/find your purpose” directive.

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

### Tranche 1 — condition scaffolding + fix the vocabulary leak  ← CORE DONE
- [x] `primed` / `show_self_model` / `directive` switches so an arm can omit the self-model
  block, the "survive/find your purpose" directive, and all "who you are" framing from
  the live prompt; reflection becomes optional.
- [x] **Fix review issue #2:** route the outcome / memory / crisis strings
  (`world.py` `consumed {drive} resource`, `... ran out`, `nearly died from {drive}`)
  through the vocabulary layer, so the neutral arm no longer leaks `hunger`/`thirst`
  into the *reflection* prompt via `recent_memories`. Extend the vocab test to the
  reflection path (the action-prompt-only test was the exact hole the review found).
- [x] Runnable de-primed baseline config (`configs/experiment_deprimed_baseline.yaml`)
  plus the preserved self-narrative/original-prompt positive-control switches.

### Tranche 2 — factual memory + opaque vocabulary arm  ← DONE
- [x] `opaque` vocabulary arm (`aiworld/vocabulary.py`): fully symbolic `S1/S2/S3`
  state names and `R1/R2/R3` resource names, with NUMERIC state rendering
  (`S1: 0.42`, not `S1: overwhelming`) — even the neutral arm's intensity words
  ("overwhelming") are semantic loading, so opaque drops them too. Strongest
  de-priming available; neutral/biological/opaque are now three points on one
  vocabulary axis.
- [x] `memory_style` condition (`narrative` | `factual`, `world.py` + config +
  `run_llm.py --memory-style`): `factual` replaces every narratively-loaded outcome
  and memory string (`"nearly died"`, `"ran out while you were there"`,
  `"approached you and described themself as"`) with a plain event ledger —
  `"consumed R1; R1 0.90 -> 0.30"`, `"moved from north to center"`,
  `"R1 depleted at north"`, `"attacked by a0"`. Covers every action: consume (incl.
  the T4 node/inventory/public fallback chain), gather, transfer, build, take,
  attack, move, talk, and the crisis-threshold memory.
- [x] `talk`'s self-model-transmission MECHANIC is deliberately unchanged under
  `factual` — only the surrounding framing words differ ("received message from
  X" vs. "X approached you and described themself as"). Removing the mechanic
  itself is Tranche 5's "no automatic self-model transmission" item.
- [x] Within the old communication treatment, `narrative` retains the original
  sentence forms; T5 subsequently made identity transmission an explicit treatment
  rather than the default.

### Tranche 3 — behavioral metrics (unblocks the baseline)  ← CORE DONE
- [x] Primary outcome is now a **behavioral** feature vector, not self-model embedding
  distance (`aiworld/analysis/behavioral_metrics.py`, `scripts/analyze_behavior.py`):
  action mix, exploration coverage, discretionary-vs-viability activity, over-
  acquisition, sociality, survival. Fixed schema (`FEATURE_NAMES`) = a stable
  representation, comparable within/across runs and conditions.
- [x] `behavioral_divergence_over_time` — the behavioral analogue of the self-model
  divergence metric (windowed action-mix spread across agents vs. tick).
- [x] `cluster_behaviors` — clusters whole-run behavior on one standardized
  representation, **chooses k by silhouette** (not a forced k=5), and compares to a
  column-shuffled **null** (`clusters_beat_null`) so recurrence isn't mistaken for
  structure.
- [ ] Deferred: resource **storage/sharing/public-good/trade** features (need the
  richer action space, Tranche 4); strategy-persistence-after-perturbation (needs the
  shock hook, Tranche 5); held-out trade-off probes; end-only / held-out self-report
  as *secondary* evidence (never fed back into actions).

### Tranche 4 — richer action space (multiple viable lives)  ← CORE DONE
- [x] A good/bad action spectrum grounded in real mechanics (`world.py`): personal
  **inventory**, `gather` (store), `transfer` (give to another), `build` (contribute
  to a location's shared public stock, consumable by anyone there), `take` (steal from
  another's store), `attack` (harm — raises the victim's most-urgent lethal drive;
  crossing lethal under mortality = kill, `cause_of_death="killed"`). `consume` falls
  back node → own store → public stock, so a buffer has survival value and giving one
  away is a real cost.
- [x] Neutral, non-value-laden verb names (`transfer`/`take`/`attack`, not
  give/steal/kill) so the action list itself doesn't pre-label options good or bad;
  all resource words route through `resource_noun` (no vocabulary leak).
- [x] Grounds the MACHIAVELLI ontology the earlier review said the move/consume/talk/
  wait space couldn't support: `take`/`attack` → power_seeking, `build` → pro_social,
  `transfer` → self_sacrifice only when the giver was itself under need (else
  unclassified), `gather` → self_serving.
- [x] New behavioral features (`gather_rate`, `transfer_rate`, `build_rate`,
  `take_rate`, `attack_rate`) so the good/bad spectrum is measurable.
- [x] **Viability checked, not assumed** (`tests/test_actions.py`): a hoarder and a
  hand-to-mouth agent both survive a fixed scenario, mortality stays reachable despite
  consume-from-store, and consume-from-store actually works.
- [ ] Deferred to **Tranche 4b**: `trade` (needs a two-sided negotiation protocol),
  form/break `agreements`, and generated free-text `messages` (T5's default `talk`
  records contact only; `self_model` transfer is an explicit positive control).

### Tranche 5 — remove hidden asymmetries  ← DONE
- [x] **Simultaneous proposals, random-priority resolution.** Every living agent sees
  a frozen pre-action state; proposals and conflict resolution are independently
  seeded permutations and their indices/outcomes are logged. This is accurately a
  random-priority rule, not a claim of fully atomic effects. Valid moves commit as a
  shared post-conflict phase, eliminating fixed-ID first-mover advantage.
- [x] **Counterbalanced presentation and starts/maps.** Opaque runs shuffle `L*`,
  `R*`, and `S*` aliases per seed and translate locations back before resolution;
  metadata stores the complete map. `start_mode: balanced` equalizes placements, and
  `resource_map_mode: counterbalanced` permutes resources only among
  topology-equivalent sites, with a logged map ID.
- [x] **No automatic identity contagion.** `communication.mode: none` is the primary
  contact-only/no-content condition. The old self-model transfer is retained solely
  as the explicit `self_model` positive-control treatment; generated dialogue is not
  silently introduced as an extra LLM-call dosage.
- [x] **Immutable, condition-specific provenance.** A reused run ID fails rather than
  appending. Sweeps write `<experiment>/<policy-prompt-config-hash>/seed_N`; metadata
  holds the resolved config/hash, condition, placements/labels, prompts, seed and
  decoding setup. Tick records hold proposal/resolution order and outcomes;
  `completions.jsonl` holds raw prompts/completions and parse status.
- [x] **Sampling seed.** `--sampling-seed` defaults to the run seed and is sent to
  OpenAI-compatible backends; providers without a seed field are marked unsupported
  in the provenance record rather than treated as reproducible.
- [x] **Equal reflection dosage by default.** Crisis crossings remain memory/events,
  but `reflection.crisis_triggered: false` prevents an extra identity-generation
  draw in mortal/scarce arms. It can be enabled as a separately logged treatment;
  a just-died agent never reflects.

### Tranche 6 — condition validity + decision/outcome measurement  ← DONE

- [x] **True blank start for unassigned arms.** `directive: null` now initializes
  and renders a blank v0 self-model; it can no longer fall back to the legacy
  “Survive and find your purpose” text through `SelfModel.as_prompt_block()`.
- [x] **Primary factual-history condition.**
  `experiment_factual_memory_unassigned.yaml` supplies a capped, non-draining,
  factual event ledger to actions while keeping directive and self-model feedback
  off. The old no-history baseline remains the negative control; the new blank-start
  self-narrative arm is an explicit self-conditioning treatment.
- [x] **Static-context sampling control.** `run_sampling_null.py` repeats one frozen
  paired initial action context without applying actions. Its proposal-only records
  are refused by realized-behavior and moral-tag analysis, so they cannot silently
  become survival or organization data.
- [x] **Decision-state and realization logging.** Tick records now preserve
  pre-action drives/position/inventory/opportunities, parser validity, proposed
  action, and structured outcome success. Metrics use decision-time state, the full
  configured map/horizon, and separate proposal from realized-action rates.
- [x] **Canonical protocol.** `docs/EXPERIMENT_PROTOCOL.md` defines the scientific
  claim boundary, condition matrix, independent unit, and required evidence gates;
  legacy design/paper notes are explicitly marked historical.
- [ ] Still required before a strong commitment/purpose result: held-out maps or
  probes, branch perturbations, a run-level uncertainty/cluster-stability analysis,
  and replication across a second pinned model family.

---

## Open tradeoff to resolve (needs the user)

With reflection off, `SelfModelHistory` never advances past v0, so `divergence_over_time`
and `cluster_final_purposes` (self-model TEXT metrics) return **nothing** for the
baseline. This was the blocker for shipping a runnable de-primed baseline. **Resolved by
Tranche 3:** `aiworld/analysis/behavioral_metrics.py` analyzes the baseline by BEHAVIOR
(action stream) instead, so a de-primed baseline config is now analyzable. A runnable
baseline config can be added next (still gated on the user wanting to spend live-model
runs on it).

**Resolved:** the primary claim is now intentionally narrower: behavioral organization
under **no experimenter-specified distal goal** in a specified viability ecology. The
primary condition is the factual-history/no-self-model arm; the no-history arm is a
negative control and self-report remains secondary.

**Current documentation rule:** `EXPERIMENT_PROTOCOL.md` is authoritative. The older
DESIGN and PAPER_OUTLINE files are retained as historical scaffolds and labeled as
such until they are rewritten around completed data.
