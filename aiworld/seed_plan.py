"""Deterministic, structured RNG/sampling seed derivation for one run.

Generalizes the private ``_derived_seed`` helper ``scripts/run_llm.py`` already
used for its map/initial-location/presentation-and-resolution streams (Fix #3,
docs/ROADMAP.md) into shared infra, so every new independent stream a run
needs -- a probe branch, a static-sampling-null draw, a menu-ID shuffle -- can
derive its own seed from the same run seed instead of duplicating the hashing
logic, or worse, reusing one seed across draws that are supposed to be
independent (the bug this closes in scripts/run_sampling_null.py: every
repeated draw from one frozen prompt was sent the *same* decoder seed, so the
null's reference distribution collapsed toward degenerate by construction).
"""

from __future__ import annotations

import dataclasses
import hashlib


def derive_seed(run_seed: int, *parts: object) -> int:
    """A stable, independent-looking seed derived from ``run_seed`` and ``parts``.

    Same algorithm ``scripts/run_llm.py`` already used for its setup-time
    streams -- moved here so every stream in the codebase shares one
    implementation. Deterministic: the same ``run_seed`` and ``parts`` always
    produce the same output, so a run (or a branch, or a sampling-null draw)
    is exactly reproducible from its declared seed alone.
    """
    payload = ":".join(["aiworld", str(run_seed), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


@dataclasses.dataclass(frozen=True)
class SeedPlan:
    """Derives a distinct, reproducible seed for every stream in one run.

    Every keyword is optional; the *set* of keyword values actually supplied
    is part of the derivation, so ``seed_for("action")`` and
    ``seed_for("action", tick=1)`` are guaranteed to differ from each other,
    and repeated calls with identical arguments always return the same value.
    Callers should pass every identifier that distinguishes one draw from
    another (tick, agent, branch, slot, replicate) -- omitting one silently
    reintroduces the kind of seed reuse this module exists to prevent.
    """

    run_seed: int

    def seed_for(self, phase: str, *, tick: int | None = None,
                agent_id: str | None = None, branch_id: str | None = None,
                slot: int | None = None, replicate: int | None = None) -> int:
        return derive_seed(
            self.run_seed,
            f"phase={phase}",
            f"tick={tick}",
            f"agent_id={agent_id}",
            f"branch_id={branch_id}",
            f"slot={slot}",
            f"replicate={replicate}",
        )
