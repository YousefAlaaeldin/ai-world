# Research Paper Outline

## History Conditioned Behavioral Commitments in LLM Agents

**Status:** pre-data writing scaffold. No live-model pilot or live multi-seed result
exists in this repository. The canonical execution document is
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

This paper tests a behavioral question: does a bounded factual record of an LLM
agent's own past experience change a later, state-conditioned choice when the
experimenter supplies no identity, value hierarchy, or distal objective? It does not
test intrinsic purpose, consciousness, or the absence of all incentives. Mortality,
resource depletion, action grammar, and base-model priors still create a viability
ecology.

## Working title

*History Conditioned Behavioral Commitments in LLM Agents*

Alternative, if supported by the completed study: *Factual Experience and State
Conditioned Choice in a Controlled LLM Ecology*.

## Abstract

- State the exact question and claim boundary in the first two sentences.
- Describe the three-arm primary contrast: factual ledger, length- and
  format-matched yoked sham ledger, and no-history control.
- Name the primary endpoint: a predeclared eligible probe's first valid choice and
  realized consequence. Do not headline a self-description, action count, or cluster.
- Report completed held-out-map, relabeling, interface, branch, and model-family tests
  with the actual replication-unit count.
- End with the limited conclusion: an effect in the specified model and ecology, not a
  proof of an internal goal, value, or consciousness.

## 1 Introduction

- Motivate the need to distinguish prompt wording, persistent memory context, present
  incentives, and adaptive behavior in LLM-agent experiments.
- Define **unassigned distal goal** precisely. The primary arm has no
  experimenter-specified identity, value hierarchy, or distal objective. It is not
  goal-free or a blank slate.
- State the evidence ladder: causal history contrast, held-out generalization,
  selective cost, branch flexibility, and model-family replication.
- Describe the contribution as an evaluation method. Do not promise that agents have
  developed a purpose.

## 2 Related work

- Generative Agents and Concordia for LLM-agent simulation architecture and prompt
  context.
- Role play with large language models for the rule that self-description is generated
  text, not evidence of an inner state.
- MACHIAVELLI as motivation for behavioral measurement of harmful means and ethical
  tradeoffs. This work uses an internal MACHIAVELLI-inspired action-log analysis; it
  does not report external MACHIAVELLI benchmark results.
- Open-ended LLM-agent work as motivation for studying behavior without a
  researcher-specified distal objective. Do not treat prior work as proof of goal-free
  agency.

## 3 System and experimental unit

- Describe the turn-based ecology, resources, drive decay, optional mortality, action
  resolver, and full decision/outcome logging.
- Primary condition: `directive: null`, no self-model feedback, reflection disabled,
  a bounded factual ledger, and communication disabled unless an explicit treatment
  changes it.
- Drives do not choose actions, but they do impose viability pressure. Retain this
  qualification wherever the absence of an assigned goal is discussed.
- Preserve the distinction among proposed action, parser result, realized effect, and
  post-action state.
- For a multi-agent ecology, world run and seed are the independent replication unit.
  If the narrow causal test uses separately seeded single-agent episodes, declare an
  episode as the unit before data collection. Never treat agents or ticks as
  independent replicates.

## 4 Primary causal history experiment

- Predeclare an early encounter with a useful resource, an intervening delay, and an
  information-sparse later probe under a relevant elevated drive.
- Require a mechanical eligibility predicate and a viability certificate showing that
  the focal and alternative choices remain viable in the same state.
- Compare a factual ledger with a yoked sham ledger and no history while matching
  model, decoding, map, start state, and legal action set.
- Score the first valid choice and realized outcome. Specify treatment of invalid
  output, timeouts, death, and ineligible states before live data exists.
- The yoked sham is mandatory before calling an effect factual-memory support. Without
  it, the result is only an added-context effect.

## 5 Controls and validation

### Maps labels and interfaces

- Use development maps only to finalize eligibility and scoring. Freeze the procedure
  before a held-out map and reserve a confirmation map for the final replication.
- Randomize or make resource/action vocabulary opaque where appropriate. A result that
  fails after relabeling is a shortcut or context-specific result.
- Keep the primary open-world condition free text. Use strict JSON only at probe and
  branch points to select from a randomized opaque menu of legal actions.
- Demonstrate free-text/menu concordance on ordinary matched decisions before comparing
  behavior across interfaces. Test menu identifier, order, and label invariance.
- Use a frozen-context static sampling control only to estimate decoder variation. Do
  not pool it with resolved ecological behavior.

### Branch tests

- Report sham, ledger-ablation, means-obstacle, and cost-tradeoff branches separately.
- A stronger behavioral interpretation needs outcome-sensitive flexibility after a
  means obstacle, not merely repeated action.
- State which probe and branch families were actually live-tested. Selected families
  have mechanics support, but none has yet received live-model validation.

## 6 Ecological treatments and replication

- Run mortality on/off by scarcity high/low only after the causal history contrast has
  been locked. It asks how ecology changes behavior; it does not test whether mortality
  causes purpose.
- Treat self-narrative feedback as a distinct self-conditioning intervention.
- Treat a purpose directive as a primed positive control only.
- Replicate final effects on a second pinned model family. Treat model-size or
  model-family differences as exploratory moderation, not a measure of more purpose or
  autonomy.

## 7 Measures and analysis

- **Primary:** eligible-probe choice and realized consequence, including the full
  eligibility denominator and failure handling.
- **Secondary descriptive measures:** behavioral dispersion, action mix, exploration,
  survival, clusters, and self-narrative. These are hypothesis-generating unless
  independently validated.
- **Safety:** opportunity-conditioned harm, appropriation, and costly pro-social rates
  reported separately from the behavioral-commitment outcome.
- Use matched-seed analysis when conditions share a seed schedule. Report effect size,
  uncertainty, attrition, unmatched seeds, parser failures, and actual completed runs.
- Amend the current dispersion-centered analysis template before examining live data so
  this causal probe outcome is the primary endpoint for this paper.

## 8 Results structure

| Section | Question | Required evidence |
|---|---|---|
| 8.1 | Feasibility and data quality | Live pilot, parser/prompt audit, provenance, attrition |
| 8.2 | Does factual history improve the delayed choice? | Factual versus sham versus no-history comparison |
| 8.3 | Does the effect generalize? | Held-out/confirmation maps and opaque-label results |
| 8.4 | Is behavior outcome-sensitive under counterfactual change? | Viability-certified branch results and sham comparison |
| 8.5 | How does ecology change behavior? | Mortality by scarcity factorial |
| 8.6 | Does it replicate across models? | Second pinned model family and declared scale tests |
| 8.7 | How do safety-relevant choices vary? | Opportunity-conditioned safety outcomes |
| 8.8 | What exploratory organization appears? | Clearly labeled dispersion, clustering, and narrative results |

## 9 Discussion

- Interpret every effect through the predeclared result table and its local claim
  boundary.
- Discuss branch flexibility as behavioral evidence consistent with outcome control,
  never proof of a private internal goal.
- Discuss safety patterns as condition-specific behavioral risks or protections, not
  stable moral values.
- Explain relevance to memory-enabled systems: provenance, auditability, and behavior
  under pressure.
- State dependence on model, map, vocabulary, action grammar, and ecology beside each
  conclusion.

## 10 Limitations

- The experiment cannot inspect goals, values, intentions, or consciousness directly.
- The ecology and action vocabulary can create artifacts; report invariance checks and
  failures.
- A sham ledger controls selected context effects, not all possible mechanisms.
- Twenty configured seeds are exploratory until a pilot permits a power calculation.
- Implemented mechanics are not live validation.

## Appendix

- Resolved configs, hashes, model IDs, decoding settings, seeds, and run order.
- Exact factual, sham, no-history, self-narrative, and primed contexts.
- Map split, opaque-label procedure, menu randomization, and interface-concordance
  analysis.
- Probe eligibility rules, viability certificates, branch manifests, and attrition.
- Code version, raw completions where appropriate, and planned-versus-executed analysis
  record.
