# Power Analysis — Blocked on Pilot Variance

**Status: not yet computable.** A real sample-size/power calculation for the
primary paired contrast (`docs/ANALYSIS_PLAN_TEMPLATE.md`) needs an estimate
of the between-seed variance of the primary outcome (`behavioral_dispersion`,
`aiworld.analysis.run_level`) under this specific model and scaffolding.
That estimate is exactly what the excluded feasibility pilot (Phase 0.5 in
this build's plan) was scoped to produce — it was deliberately left out of
this infrastructure build (a competing process had the GPU/CPU at build
time), so this document states the method and leaves the actual numbers for
after that pilot runs. Do not fill in placeholder numbers here to make this
document look complete; an invented variance estimate is worse than an
honestly blocked one.

## Method (to run once pilot variance exists)

1. From the pilot's small seed set (5-10 seeds is enough to ESTIMATE
   variance, not to test anything), compute per-seed `behavioral_dispersion`
   for the primary condition (`aiworld.analysis.run_level.run_summary`).
2. Estimate the standard deviation of that outcome across seeds, `sigma_hat`.
3. Decide a minimum effect size worth detecting, `delta` — a difference in
   `behavioral_dispersion` between two conditions that would be
   scientifically meaningful, not merely statistically detectable. This is a
   judgment call the pilot's data cannot make for you; state it explicitly
   before computing power, not after seeing what the data can distinguish.
4. Solve for the seed count needed at a chosen alpha (0.05 conventional) and
   power (0.8 conventional) — either a standard paired-difference power
   formula, or a simulation that reuses
   `aiworld.analysis.run_level.paired_condition_comparison`'s own
   bootstrap/permutation machinery against synthetic data built from
   `sigma_hat` and a range of `delta` and seed counts.
5. Compare the result against the 20-seed schedule already declared in
   `configs/experiment_factual_memory_unassigned.yaml` and its siblings,
   which is explicitly labeled there as exploratory, not powered. Raise the
   schedule if the computed requirement exceeds 20; lower it (with
   justification) only if the pilot shows tight, well-behaved variance.

## What not to do

Do not report a power calculation using a variance estimate assumed, guessed,
or borrowed from a different model, prompt scaffold, or published result.
Variance here is a property of THIS model under THIS action grammar and
prompt scaffold — the same claim-boundary discipline
`docs/EXPERIMENT_PROTOCOL.md` applies to result language applies to power-
analysis inputs too.
