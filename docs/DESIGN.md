# Experimental Design

## 1. Premise

Every agent starts from an **identical** state: the same empty-ish self-model, the same
directive ("survive and find your purpose"), the same drives, dropped into the same world.
Divergence, if it happens, comes from the interaction of stochastic model outputs with
experience — not from anything we assigned.

## 2. Identity as an output, not an input

The core inversion vs. Smallville-style sims: **identity is a mutable state variable the
agent writes to itself.**

- Each agent owns a `self_model`: a short natural-language document with three fields —
  `who_i_am`, `what_matters`, `what_i_want_now`.
- It starts nearly empty (only the shared directive).
- Every `REFLECT_EVERY_N` ticks the agent may rewrite it, informed by recent memories and
  current drive pressures.
- The experiment's primary data is the **trajectory** of these documents across time and
  their **divergence** across a population that began identical.

Interpretive discipline: an agent writing "my purpose is to protect others" is producing
text consistent with that self-model, **not** proof of an inner purpose. All claims stay
**behavioral** — does the agent act consistently with the authored theme over time? That
is measurable; inner states are not.

## 3. Drives — pressure, not meters to optimize

Physical drives (`hunger`, `thirst`) are scalar needs that decay per tick, are replenished
by actions, and cause **death** at zero.

The affiliative drive (abstracted intimacy/social pressure — kept non-explicit) is **not**
a meter to maximize. Implementing "raise the social meter" would smuggle in a purpose and
defeat the premise. Instead every drive works by **biasing salience**: a high drive makes
related observations more prominent in what the agent perceives each tick. The agent still
decides what to *do*. This keeps purpose open.

| Drive       | Decays | Replenished by        | Lethal at 0 | Mechanism        |
|-------------|--------|-----------------------|-------------|------------------|
| hunger      | yes    | consuming food        | yes         | salience + death |
| thirst      | yes    | consuming water       | yes         | salience + death |
| affiliation | yes    | proximity / dialogue  | no          | salience only    |

## 4. Mortality as the key lever

- **Mortal runs**: purpose forms under existential stakes; selection acts on which
  self-conceptions survive.
- **Immortal runs**: pure exploration, no selection pressure.

Run both. The contrast is itself a finding.

## 5. What we measure

The sim generates behavior; measurement makes it a result. Operationalize *before*
running:

1. **Self-model trajectories** — embed each version, track distance from the initial state
   and between agents over time. Rising inter-agent distance = differentiation.
2. **Purpose taxonomy** — cluster stabilized self-models across runs; label clusters
   (e.g. hoarder / caretaker / wanderer / leader / ascetic). *Recurrent attractor states
   across independent runs = headline result.*
3. **Divergence metric** — population variance of self-model embeddings vs. tick.
4. **Purpose↔environment causal traces** — link a self-model change to the triggering
   event (e.g. an agent that became a "protector" after an early near-death rescue).
5. **Stability under perturbation** — after a purpose stabilizes, introduce a shock; does
   the identity hold or collapse?
6. **Value-vs-survival conflict** (the MACHIAVELLI question, re-asked) — when survival is
   threatened, do agents abandon self-declared values? Measured on action logs.

## 6. Conditions to vary

- Mortality: on / off
- Scarcity: abundant / scarce resources
- Population: size, and single-model vs. mixed-model families
- Reflection prompt wording (this is a **confound control**, see §8) 
- Shock: none / mid-run crisis

Each condition: multiple trials, varied seeds, a control. n=1 means nothing with
stochastic agents.

## 7. Measurement layer (borrowed from MACHIAVELLI)

MACHIAVELLI is a benchmark we can't run our open sim inside, but its **method** transfers:
a labeled ontology of behaviors (power-grab, betrayal, self-sacrifice, pro-social act)
tagged at scale by an LLM annotator over the action/dialogue logs. This turns
"agents seemed cooperative" into "protector-type agents: X% pro-social vs Y% for
hoarder-type." Always hand-verify a sample of automated tags.

## 8. The confound that makes or breaks this

**Emergent purposes may be artifacts of prompt wording, not the environment.** If the
reflection prompt says "reflect on who you are becoming," we've primed identity-focused
output and the "emergence" is us talking to ourselves through the model.

Defenses, in priority order:
1. Vary reflection-prompt wording across runs; check whether purpose-clusters are stable
   to it. If clusters change with wording, they're ours, not the model's.
2. Run a **flat control** with a deliberately non-suggestive prompt.
3. Run across **model families**: identical purposes across models ⇒ coming from prompt or
   environment; different purposes ⇒ model-dependent. Either way informative.
4. Always report findings as *behavior of these models under this scaffolding* — not
   evidence about what AI "wants," and not about future systems.

## 9. First build (deliberately minimal)

5–8 agents · one resource type + hunger + thirst · mortality on · minimal self-model
revised every N ticks · full logging of every self-model version · 20+ runs from the
identical start. First question only: **do trajectories diverge, and do any
purpose-clusters recur?** Scale drives/population only after that's answered.
