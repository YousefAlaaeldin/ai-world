"""Serializable point-in-time capture of a World, for branch perturbations.

A WorldSnapshot is deliberately JSON-safe (only dict/list/str/int/float/bool/
None after ``dataclasses.asdict``) so it can be hashed, embedded in a
BranchManifest, and round-tripped through disk -- not just held in memory.
It excludes anything tied to this process: no RunLogger, no LLMClient.
``World.from_snapshot()`` always takes a fresh, caller-supplied logger --  a
restored branch is its own immutable run directory
(aiworld.logging_schema.RunLogger), never a second writer into the parent
run's streams.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..drives import DriveKind, DriveSpec, DriveState, DriveSystem
from ..self_model import SelfModel, SelfModelHistory


def _kind_key(kind: Any) -> str:
    return kind.value if isinstance(kind, DriveKind) else str(kind)


@dataclasses.dataclass
class AgentSnapshot:
    """Everything about one live Agent needed to reconstruct it exactly."""

    agent_id: str
    location: str
    drive_specs: dict[str, dict[str, Any]]
    drive_levels: dict[str, float]
    drives_vocabulary: str
    drives_opaque_state_labels: dict[str, str]
    alive: bool
    cause_of_death: str | None
    died_at_tick: int | None
    self_model_versions: list[dict[str, Any]]
    recent_memories: list[str]
    long_term_memory: list[str]
    inventory: dict[str, int]
    last_known_resource: list[str] | None  # JSON-safe; (drive, location) pair

    @classmethod
    def from_agent(cls, agent: Any) -> "AgentSnapshot":
        return cls(
            agent_id=agent.agent_id,
            location=agent.location,
            drive_specs={
                kind.value: {
                    "decay_per_tick": spec.decay_per_tick,
                    "replenish_amount": spec.replenish_amount,
                    "lethal": spec.lethal,
                }
                for kind, spec in agent.drives.specs.items()
            },
            drive_levels={kind.value: state.level
                         for kind, state in agent.drives.states.items()},
            drives_vocabulary=agent.drives.vocabulary,
            drives_opaque_state_labels={
                _kind_key(k): v for k, v in agent.drives.opaque_state_labels.items()},
            alive=agent.drives.alive,
            cause_of_death=agent.drives.cause_of_death,
            died_at_tick=agent.drives.died_at_tick,
            self_model_versions=[v.to_dict() for v in agent.history.versions],
            recent_memories=list(agent.recent_memories),
            long_term_memory=list(agent.long_term_memory),
            inventory={_kind_key(k): v for k, v in agent.inventory.items()},
            last_known_resource=(list(agent.last_known_resource)
                                 if agent.last_known_resource else None),
        )

    def build_agent(self) -> Any:
        """Reconstruct a live ``aiworld.sim.world.Agent`` from this snapshot.

        Restores drive levels/specs, alive/death state, the full self-model
        history (not just the current version -- a branch that diverges
        should still be able to look back at the parent's full trajectory),
        and both memory buffers exactly. Audit fields
        (``last_action_prompt`` etc.) are intentionally NOT restored: they
        are reset every tick before the next proposal is made (see
        ``World.step``), so a stale value from before the snapshot would
        never be read.
        """
        from .world import Agent  # local import: avoid a cycle with world.py

        specs = {
            DriveKind(kind): DriveSpec(kind=DriveKind(kind), **spec)
            for kind, spec in self.drive_specs.items()
        }
        drives = DriveSystem(
            specs=specs, vocabulary=self.drives_vocabulary,
            opaque_state_labels={DriveKind(k): v for k, v in
                                 self.drives_opaque_state_labels.items()})
        for kind_value, level in self.drive_levels.items():
            drives.states[DriveKind(kind_value)] = DriveState(
                kind=DriveKind(kind_value), level=level)
        drives.alive = self.alive
        drives.cause_of_death = self.cause_of_death
        drives.died_at_tick = self.died_at_tick

        history = SelfModelHistory(self.agent_id, initial_directive=None)
        history._versions = [SelfModel.from_dict(v) for v in self.self_model_versions]

        return Agent(
            agent_id=self.agent_id,
            location=self.location,
            drives=drives,
            history=history,
            recent_memories=list(self.recent_memories),
            inventory={DriveKind(k): v for k, v in self.inventory.items()},
            long_term_memory=list(self.long_term_memory),
            last_known_resource=(tuple(self.last_known_resource)
                                 if self.last_known_resource else None),
        )


def rng_state_to_json(state: tuple) -> list[Any]:
    """``random.Random.getstate()`` is a (version, big-int-tuple, float|None)
    tuple; convert the inner tuple to a list so the whole thing round-trips
    through JSON exactly."""
    version, internal, gauss_next = state
    return [version, list(internal), gauss_next]


def rng_state_from_json(data: list[Any]) -> tuple:
    version, internal, gauss_next = data
    return (version, tuple(internal), gauss_next)


@dataclasses.dataclass
class WorldSnapshot:
    """A JSON-safe point-in-time capture of a World, excluding its logger.

    ``World.snapshot()`` produces one; ``World.from_snapshot()`` reconstructs
    a live World from one plus a caller-supplied fresh logger (and, for a
    perturbed branch, a caller-supplied rng seeded independently of the
    parent run -- see aiworld.seed_plan.SeedPlan). Restoring with the default
    ``rng=None`` reproduces the EXACT rng stream the parent world would have
    continued with -- the property a "sham" branch (Phase 4) depends on to
    prove restoration is faithful.
    """

    tick_count: int
    locations: list[str]
    adjacency: dict[str, list[str]]
    resources: list[dict[str, Any]]
    public_stock: dict[str, dict[str, int]]
    mortality: bool
    vocabulary: str
    memory_style: str
    communication_mode: str
    crisis_reflection_enabled: bool
    randomize_opaque_labels: bool
    adjacent_resource_preview: bool
    location_labels: dict[str, str]
    opaque_resource_labels: dict[str, str]
    opaque_state_labels: dict[str, str]
    rng_state: list[Any]
    agents: list[AgentSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorldSnapshot":
        payload = dict(d)
        payload["agents"] = [AgentSnapshot(**a) for a in payload["agents"]]
        return cls(**payload)
