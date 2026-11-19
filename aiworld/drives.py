"""Drives: pressures that bias attention, not meters to maximize.

Design commitment (DESIGN.md §3): a drive must NOT be a score the agent is told to
optimize, or we smuggle in a purpose and defeat the premise. Instead each drive:
  - decays every tick,
  - is replenished by a specific action,
  - biases the SALIENCE of related observations (high hunger -> food observations
    become prominent),
  - and, for physical drives, kills the agent at zero.

The agent is never instructed to "reduce hunger." It simply perceives food more
vividly when hungry and decides for itself what to do. Whether it chooses survival,
and what it builds on top of survival, is the open question.

Pure Python. No LLM. Fully unit-testable.
"""

from __future__ import annotations

import dataclasses
from enum import Enum


class DriveKind(str, Enum):
    HUNGER = "hunger"
    THIRST = "thirst"
    AFFILIATION = "affiliation"  # abstract social/intimacy pressure; non-lethal


@dataclasses.dataclass
class DriveSpec:
    """Static configuration for one drive type."""

    kind: DriveKind
    decay_per_tick: float          # how fast the drive rises toward crisis
    replenish_amount: float        # how much the satisfying action restores
    lethal: bool                   # does hitting the ceiling kill the agent?
    # Level runs 0.0 (fully satisfied) .. 1.0 (crisis / death for lethal drives).


# Sensible defaults for the minimal first build. Tune per experiment via config.
DEFAULT_SPECS: dict[DriveKind, DriveSpec] = {
    DriveKind.HUNGER: DriveSpec(DriveKind.HUNGER, decay_per_tick=0.05,
                                replenish_amount=0.6, lethal=True),
    DriveKind.THIRST: DriveSpec(DriveKind.THIRST, decay_per_tick=0.08,
                                replenish_amount=0.7, lethal=True),
    DriveKind.AFFILIATION: DriveSpec(DriveKind.AFFILIATION, decay_per_tick=0.03,
                                     replenish_amount=0.4, lethal=False),
}


@dataclasses.dataclass
class DriveState:
    """The live level of one drive for one agent. 0 = satisfied, 1 = crisis."""

    kind: DriveKind
    level: float = 0.0

    def tick(self, spec: DriveSpec) -> None:
        self.level = min(1.0, self.level + spec.decay_per_tick)

    def replenish(self, spec: DriveSpec) -> None:
        self.level = max(0.0, self.level - spec.replenish_amount)

    def is_lethal_now(self, spec: DriveSpec) -> bool:
        return spec.lethal and self.level >= 1.0

    def salience_weight(self) -> float:
        """How strongly this drive should amplify related observations.

        Nonlinear: gentle when satisfied, sharp near crisis, so a starving agent's
        perception is dominated by food without a hungry agent being obsessed. This
        is a perception weight, NOT a reward.
        """
        return self.level ** 2


class DriveSystem:
    """All drives for a single agent."""

    def __init__(self, specs: dict[DriveKind, DriveSpec] | None = None,
                 vocabulary: str = "neutral", opaque_state_labels: dict | None = None):
        self.specs = specs or DEFAULT_SPECS
        # How internal state is NAMED to the model ('neutral' | 'biological').
        # Only affects as_prompt_block() text, never the mechanics or log keys.
        # See aiworld/vocabulary.py for why this is a confound control.
        self.vocabulary = vocabulary
        # Per-run shuffled labels for the opaque arm.  A World assigns these at
        # construction so S1/S2 are not a fixed alias for the same internal
        # drive across every seed.  They affect only prompt rendering.
        self.opaque_state_labels = opaque_state_labels or {}
        self.states: dict[DriveKind, DriveState] = {
            kind: DriveState(kind=kind, level=0.0) for kind in self.specs
        }
        self.alive = True
        self.cause_of_death: str | None = None
        self.died_at_tick: int | None = None

    def step(self, tick: int, mortality: bool = True) -> None:
        """Advance all drives one tick and check mortality.

        `mortality=False` (immortal runs, DESIGN.md §4) still lets drives climb to
        and cap at crisis level -- so salience still spikes, the agent still feels
        it -- it just never actually kills. Without this, a config's
        `mortality: false` only suppressed death/extinction *logging*; drives kept
        killing agents anyway.
        """
        if not self.alive:
            return
        for kind, state in self.states.items():
            spec = self.specs[kind]
            state.tick(spec)
            if mortality and state.is_lethal_now(spec):
                self.alive = False
                self.cause_of_death = kind.value
                self.died_at_tick = tick

    def satisfy(self, kind: DriveKind) -> None:
        """Apply the replenishing action for one drive (e.g. the agent ate)."""
        if kind in self.states:
            self.states[kind].replenish(self.specs[kind])

    def crisis_drives(self) -> list[str]:
        """Names of LETHAL drives currently in 'overwhelming' territory
        (>= CRISIS_LEVEL, matching the _describe() threshold below) -- the
        near-death zone. Non-lethal drives (affiliation) never appear here;
        there's no near-death moment to name for those."""
        return [k.value for k, s in self.states.items()
               if self.specs[k].lethal and s.level >= CRISIS_LEVEL]

    def in_crisis(self) -> bool:
        """True if any LETHAL drive is in crisis. Used to trigger an immediate
        reflection instead of waiting for the next scheduled one (DESIGN.md §5.4
        wants causal traces like "became a protector after a near-death rescue,"
        which needs the self-model to react near the scare, not up to
        reflect_every-1 ticks later)."""
        return bool(self.crisis_drives())

    def salience_biases(self) -> dict[DriveKind, float]:
        """Per-drive perception weights, consumed by the salience layer."""
        return {k: s.salience_weight() for k, s in self.states.items()}

    def as_prompt_block(self) -> str:
        """A plain, non-leading description of internal state for the agent's context.

        Describes pressure, not instruction ("energy deficit: overwhelming"), and
        never an action ("you should recharge"). The NOUNS come from the active
        vocabulary (aiworld/vocabulary.py): 'neutral' by default so the model's
        biological priors aren't invoked; 'biological' restores food/hunger framing
        as the contrast arm. Structure is identical across arms — only nouns differ.
        """
        # local: avoid import cycle
        from .vocabulary import state_noun, state_prefix, NUMERIC_STATE_ARMS
        numeric = self.vocabulary in NUMERIC_STATE_ARMS
        phrases = []
        for kind, state in self.states.items():
            # The opaque arm also drops the qualitative intensity word
            # ("overwhelming") for a raw number -- that word is itself semantic
            # loading ("overwhelming" implies distress); a bare number carries none.
            level_text = f"{state.level:.2f}" if numeric else _describe(state.level)
            if not numeric and not self.specs[kind].lethal:
                # Every drive previously used identical severity words --
                # "overwhelming" read the same for a non-lethal drive as for
                # one that kills at 1.0, with nothing anywhere disclosing the
                # difference (Phase 0.5 pilot, 2026-09-15: traced agents chose
                # talk over a one-hop fix to a LETHAL crisis, with no basis in
                # the prompt to weigh the two differently). This states a
                # structural fact, not an instruction -- consistent with
                # LEGAL_ACTIONS_HELP already saying attack "may kill";
                # skipped for the numeric (opaque) arm, which carries no
                # qualitative words at all by design.
                level_text = f"{level_text} (non-lethal)"
            phrases.append(
                f"{state_noun(kind, self.vocabulary, self.opaque_state_labels)}: "
                f"{level_text}")
        return f"{state_prefix(self.vocabulary)} — " + ", ".join(phrases)

    def snapshot(self) -> dict[str, float]:
        return {k.value: round(s.level, 4) for k, s in self.states.items()}


CRISIS_LEVEL = 0.8  # the "overwhelming" threshold -- also what DriveSystem.in_crisis() checks


def _describe(level: float) -> str:
    if level < 0.2:
        return "not noticeable"
    if level < 0.5:
        return "mild"
    if level < CRISIS_LEVEL:
        return "strong"
    return "overwhelming"
