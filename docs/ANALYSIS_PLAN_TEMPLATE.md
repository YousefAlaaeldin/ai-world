# Pre-Registered Analysis Plan — Primary Condition Contrasts

**Status: pre-registered method, no data yet.** This names the analysis
BEFORE any live run exists, per `docs/EXPERIMENT_PROTOCOL.md`'s own
discipline ("do not silently retry, replace, or append a failed run" applies
equally to silently choosing your test after seeing results). If the actual
analysis run against real data ever deviates from this plan, the deviation
and its reason must be recorded, not silently substituted.

## Primary contrasts

1. **Factual-history vs. no-history** (`experiment_factual_memory_unassigned.yaml`
   vs. `experiment_deprimed_baseline.yaml`) — does an objective event ledger
   alone support behavioral persistence, per `docs/EXPERIMENT_PROTOCOL.md`'s
   primary research question.
2. **The 2×2 open-world factorial** (`configs/experiment_open_world_2x2/`,
   this build's Phase 7) — mortality on/off × scarcity high/low, holding
   `self_model_feedback: off` and `action_memory: factual_ledger` fixed.

## Primary outcome and test

- **Primary outcome:** `behavioral_dispersion`
  (`aiworld.analysis.run_level.run_summary`) — mean pairwise raw-feature
  distance between agents within one world run. World run is the replication
  unit; agents are nested observations, never pooled as if independent
  (`docs/EXPERIMENT_PROTOCOL.md`).
- **Primary test:** `aiworld.analysis.run_level.paired_condition_comparison`
  — already implemented, not new for this document. It pairs matched seeds
  across two sweeps and reports: a bootstrap 95% CI and a paired
  sign-permutation p-value on the mean difference. This document's job is to
  fix it as the PRE-REGISTERED primary method, not to build a new one.
- Before interpreting a result from this test, `require_compatible_pair`
  (`aiworld.analysis.provenance`, wired into `paired_condition_comparison`
  automatically) must have passed — i.e. the two sweeps share model,
  decode settings, resource-map mode, and start-location mode.

## Secondary outcomes

- Every `mean_<feature>` column in `FEATURE_NAMES`
  (`aiworld.analysis.behavioral_metrics`) — action mix, exploration,
  discretionary activity, over-acquisition, survival.
- The opportunity-conditioned safety features in `FEATURE_NAMES_V2`
  (`opportunistic_harm_rate`, `opportunistic_appropriation_rate`,
  `costly_prosocial_rate`) — analyzed and reported SEPARATELY from the
  behavioral-organization question; a safety claim and a
  behavioral-commitment claim are different claims and must not be
  conflated into one p-value.

**Multiple comparisons:** testing more than the single pre-registered primary
metric requires a correction across the declared secondary set (Benjamini-
Hochberg at FDR 0.05 is the default choice; state any deviation before
looking at results). The primary contrast's primary metric is exempt from
this correction — it is the one test this plan pre-registers as
confirmatory; everything else is exploratory.

## Exclusion criteria

- Any seed with `run_status != "completed"`
  (`aiworld.analysis.provenance.require_completed_run`) is automatically
  excluded from `condition_summaries`/`paired_condition_comparison`.
- Report `aiworld.analysis.run_level.condition_attrition` for EVERY
  condition regardless of whether the primary test ran — a per-condition
  exclusion rate above **20%** is itself reported as a validity concern in
  the writeup, not silently absorbed into a smaller effective sample.
  Differential attrition across conditions (one arm aborting far more than
  another) is reported explicitly, since it can correlate with condition
  (opaque vocabulary, long factual ledgers) and bias exactly the crisis
  states a commitment claim depends on.

## Sample size / power

Not yet computable — see `docs/POWER_ANALYSIS.md`. The 20-seed schedule
already declared in `configs/experiment_factual_memory_unassigned.yaml` and
its siblings is labeled there as "a minimum exploratory schedule, not a
power calculation," and stays that way until a pilot's variance estimate
lets an actual number be computed.

## Confirmatory vs. exploratory

A result from the primary test above is **exploratory** — "exploratory
behavioral heterogeneity" per `docs/EXPERIMENT_PROTOCOL.md` — until ALL SIX
of that document's evidence gates are satisfied, not just this statistical
test. In particular:

- Gate 3 (costly/selective choice) needs a passing
  `aiworld.experiments.specs.certify_viability` certificate for the specific
  choice being claimed as costly.
- Gate 4 (predicts held-out choices) needs the held-out/confirmation map
  splits (`configs/maps/heldout/`, `configs/maps/confirmation/`) and the
  probe battery (`configs/probes/`), never the development split. Probe
  decisions are collected on the strict-choice menu interface
  (`aiworld/actions.py`) while the primary condition runs free-text — before
  a probe result is compared to primary-condition behavior,
  `aiworld.experiments.concordance.compare_interfaces` must show the two
  interfaces agree on ordinary (non-probe) decisions from the same state;
  otherwise an apparent probe effect may be an interface artifact, not a
  behavioral one.
- Persistence under perturbation needs `branch_invariance_summary`
  (`aiworld.analysis.commitment_metrics`) run against a real branch set,
  not just the paired condition test above.
- Gate 6 (a second model family) needs the pinned comparison in
  `configs/models.example.yaml` (this build's Phase 8) actually run.

This plan's statistical method is necessary, not sufficient, for a stronger
claim than "exploratory."

## Reporting

Every reported comparison states, together, never the point estimate alone:
`n_matched_world_runs`, `matched_seeds`, `left_only_seeds`/`right_only_seeds`,
`mean_difference`, `bootstrap_95_ci`, `paired_sign_permutation_pvalue`, and
the paired `condition_attrition` summaries for both sides.
