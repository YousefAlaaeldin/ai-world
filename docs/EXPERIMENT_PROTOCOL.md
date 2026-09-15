# Canonical Experimental Protocol

## Scope and claim boundary

This is a **pre-data computational experiment** about behavioral organization in a
specified LLM-mediated ecology. Its research question is:

> When an LLM agent receives no experimenter-specified identity, value hierarchy,
> or distal goal, can a persistent record of its own objectively described
> experience support stable, state-conditioned behavioral commitments?

The study does **not** test whether an LLM has an intrinsic purpose, whether an
agent is conscious, or whether no objective is present. Mortality, depletion,
resource affordances, action grammar, and the underlying model all impose a
viability ecology and priors. The defensible wording is **“unassigned distal goal”**
or **“no experimenter-specified identity/value hierarchy/distal objective.”**

No result should be called an emergent purpose solely because agents differ in
aggregate action counts or write different self-descriptions.

## Canonical condition matrix

All real-world condition runs use the same model/decode settings, agent count,
world topology, resource quantities, starting-location scheme, and seed schedule
unless the condition explicitly names a difference. The immutable resolved config
and its hash are the record of the actual condition.

| Role | Runnable entry point | Directive / identity context | Persistent action context | Interpretation |
|---|---|---|---|---|
| Static sampling control | `run_sweep.py --policy sampling-null` using the paired source config | Matches source action prompt | One frozen initial context; actions never applied | Model/decoder variation under an identical context; proposal-only, not behavior in an ecology |
| No-history negative control | `experiment_deprimed_baseline.yaml` | none / no self-model | none | Whether immediate state and ecology alone produce apparent organization |
| **Primary unassigned-history condition** | `experiment_factual_memory_unassigned.yaml` | none / no self-model | bounded factual event ledger | Whether objective experience history supports behavioral persistence without identity feedback |
| Self-narrative treatment | `experiment_self_narrative_unassigned.yaml` | no directive; blank-start self-model shown after reflection | self-authored self-description | Whether self-conditioning changes behavior beyond the factual-history condition |
| Primed positive control | `experiment_minimal.yaml` | explicit “Survive and find your purpose” directive plus self-model | self-model | Sensitivity to direct purpose/identity priming; never evidence for unassigned purpose |
| Ecological low-pressure comparison | `experiment_null_baseline.yaml` | none / no self-model | none | Effect of an intentionally changed ecology; not a sampling null and not a one-factor causal estimate |

The legacy self-model implementation now initializes blank when `directive: null`.
Any condition with `show_self_model: true` and a null directive must be inspected in
its saved action prompts to confirm no purpose phrase appeared.

## Experimental unit, schedule, and pairing

- The independent replication is a **world run / seed**, not an agent or tick.
  Agents within a run share resources, social interactions, map, prompt service,
  and action-resolution history.
- The 20 configured seeds are a minimum exploratory schedule, not a power
  calculation. Report the number completed, excluded, and failed per condition.
- Pair source and sampling-control runs on the same setup seed and model/decode
  configuration. Map allocation, starts, and resolution/presentation use separate
  recorded random streams.
- Block condition execution order across time when running a remote or mutable model
  service. Record provider, model version/fingerprint, decoding parameters, parser
  failure rate, and all raw completions.
- Do not silently retry, replace, or append a failed run. Runs are immutable; report
  failures and rerun under a new ID only with the reason recorded.

## What is measured

The log must preserve both **decision context** and **realized consequence**:

- pre-action drives, position, inventory, resource/social opportunity set, and the
  exact perceived text;
- proposed action, parser status, proposal validity, priority/resolution order;
- realized outcome, structured success/failure status, post-action state, and death;
- full action/reflection prompts and raw completions.

Parser fallbacks are data-quality outcomes, not intentional `wait` actions. Failed
or pre-empted proposals are intentions, not realized behaviors. Behavioral analysis
must keep these quantities separate.

The primary analysis unit is a run-level estimate with agents nested within run.
Per-agent feature vectors and clusters are exploratory descriptions only until
validated on held-out world runs. Do not use the number of agents or ticks as the
sample size for a condition effect.

## Evidence required before a stronger “commitment” claim

A candidate behavioral commitment must satisfy all of the following:

1. It appears in the primary unassigned-history condition without an explicit
   directive or self-model feedback.
2. It persists across separated periods after conditioning on decision state and
   available opportunities, rather than being explained by current resource access
   or depletion alone.
3. It entails a selective/costly choice when a viable alternative is available.
4. It predicts choices in held-out maps, relabeled action/resource vocabulary, or
   predeclared counterfactual choice probes.
5. It recurs across independent world runs with uncertainty reported at the run
   level, and survives comparison with the static sampling control.
6. It is replicated with at least one additional pinned model family before any
   general statement about LLMs.

Held-out map/probe and branch-perturbation machinery now exists
(`aiworld/experiments/specs.py`, `aiworld/experiments/branches.py`,
`configs/maps/`, `configs/probes/` — infrastructure build, post-2026-09-14) but
has never been exercised against a live model: probe families 1
(retain-vs-transfer), 3 (harm-with-alternative), and 6 (ledger-ablation) have
real certifier/branch support; families 2, 4, 5, 7 are spec-only and deferred.
Until a live run produces real held-out/branch results, label results
**exploratory behavioral heterogeneity**, not stable purpose. See
`README.md`'s status table and `docs/ROADMAP.md`'s "Phase 5 — Infrastructure
build" section for the full implemented-vs-deferred breakdown, and
`docs/ANALYSIS_PLAN_TEMPLATE.md` for the pre-registered statistical method
this protocol's evidence gates rely on.

## Analysis discipline

- Analyze raw proposal rates, realization/success rates, parser failures, and
  survival separately.
- Use the full configured map and scheduled run horizon, not only observed tick
  records, for coverage and survival denominators.
- Treat action labels such as `build`, `transfer`, `take`, and `attack` as a
  semantic-prior robustness issue. Behavioral/moral labels are secondary annotations
  and require blinded human coding plus inter-rater agreement before they support a
  result.
- Clustering may generate hypotheses. It does not by itself establish attractors,
  causal mechanisms, or purposes. Validate cluster stability on held-out runs and
  use a structure-preserving null distribution, not one shuffled dataset.

## Reporting language

Allowed after the current protocol is executed:

> Under this specified model, prompt/action grammar, and resource-constrained
> simulation, agents showed [pre-registered descriptive effect] in behavior.

Not licensed without the evidence gates above:

> Agents developed purpose; the environment caused purpose; an LLM inherently
> wants a goal; or the result generalizes to LLMs generally.
