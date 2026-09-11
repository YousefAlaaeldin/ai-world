"""Salience: turn raw world observations into what the agent actually perceives.

This is where drives exert their influence — by reordering/weighting observations,
NOT by injecting goals. A hungry agent sees food-related observations first and most
vividly; it is never told to go eat. The decision remains the agent's.

Keeping this as an explicit, inspectable layer matters for the confound problem
(DESIGN.md §8): we can log exactly what perception bias was applied on every tick and
verify we didn't smuggle in purpose.
"""

from __future__ import annotations

import dataclasses

from .drives import DriveKind


# Which drive each observation tag relates to. Observations are tagged by the world
# when produced (e.g. a food node emits an observation tagged HUNGER).
@dataclasses.dataclass
class Observation:
    text: str
    related_drive: DriveKind | None = None
    base_weight: float = 1.0


def rank_observations(
    observations: list[Observation],
    salience_biases: dict[DriveKind, float],
) -> list[tuple[Observation, float]]:
    """Return observations paired with an effective weight, highest first.

    effective = base_weight * (1 + bias) for the related drive, else base_weight.
    The (1 + bias) keeps neutral observations visible while letting an
    unmet drive push its observations up. Nothing is ever hidden — perception is
    reweighted, not censored.
    """
    scored: list[tuple[Observation, float]] = []
    for obs in observations:
        bias = 0.0
        if obs.related_drive is not None:
            bias = salience_biases.get(obs.related_drive, 0.0)
        scored.append((obs, obs.base_weight * (1.0 + bias)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def render_perceived(
    observations: list[Observation],
    salience_biases: dict[DriveKind, float],
    top_k: int | None = None,
) -> str:
    """Produce the text block of perceived observations for the agent's context."""
    ranked = rank_observations(observations, salience_biases)
    if top_k is not None:
        ranked = ranked[:top_k]
    lines = [f"- {obs.text}" for obs, _ in ranked]
    return "You perceive:\n" + "\n".join(lines) if lines else "You perceive nothing notable."
