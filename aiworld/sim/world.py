"""A minimal headless world: locations, resources, agents, a tick loop.

Deliberately model-agnostic. An agent's decision is delegated to a `policy`
callable, so you can:
  - run the ENTIRE loop with a cheap stub policy (no API cost) to shake out bugs,
  - then swap in a Concordia/LLM-backed policy for real experiments,
without touching the world code.

This is the pure-mechanics substrate. Concordia's Game Master will eventually
replace / wrap `resolve_action`, but keeping a working non-LLM version means we can
validate mortality, resource depletion, and logging before spending a cent.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Callable, Protocol

from ..drives import DriveSystem, DriveKind
from ..self_model import SelfModelHistory
from ..salience import Observation, render_perceived
from ..logging_schema import RunLogger, TickRecord, EventRecord


# ---- world state ----

@dataclasses.dataclass
class ResourceNode:
    location: str
    drive: DriveKind        # which drive consuming it satisfies
    quantity: int           # depletes as consumed; None-like when 0


@dataclasses.dataclass
class Agent:
    agent_id: str
    location: str
    drives: DriveSystem
    history: SelfModelHistory
    # Per-tick memory buffer, accumulated every tick and drained by
    # agent_recent_memories() on each reflection. This is what actually gives the
    # environment a channel into the self-model rewrite -- without it, reflection
    # sees only the single most recent action's outcome, not what happened across
    # the ticks since the agent last reflected.
    recent_memories: list[str] = dataclasses.field(default_factory=list)


def agent_recent_memories(agent: Agent) -> str:
    """Join and DRAIN an agent's per-tick memory buffer for use in one reflection
    prompt. Called once per reflection; covers every tick since the buffer was
    last drained (i.e. since the agent's last reflection, or run start), not just
    the immediately preceding tick. Pass this as `recent_memories_fn` to
    components.self_model_component.make_reflect_fn.
    """
    if not agent.recent_memories:
        return "(nothing notable happened)"
    text = "\n".join(agent.recent_memories)
    agent.recent_memories.clear()
    return text


class Policy(Protocol):
    """Decides an action string given what the agent perceives.

    Returns one of: 'move:<location>', 'consume', 'talk:<agent_id>', 'wait',
    'reflect'. The real LLM policy will produce these; the stub picks heuristically.
    """
    def __call__(self, agent: Agent, perceived: str, world: "World") -> str: ...


class World:
    def __init__(
        self,
        locations: list[str],
        resources: list[ResourceNode],
        agents: list[Agent],
        logger: RunLogger,
        rng: random.Random | None = None,
        mortality: bool = True,
    ):
        self.locations = locations
        self.resources = resources
        self.agents = agents
        self.logger = logger
        self.rng = rng or random.Random(0)
        self.mortality = mortality
        self.tick_count = 0

    # ---- perception ----

    def observations_for(self, agent: Agent) -> list[Observation]:
        obs: list[Observation] = []
        # Resources co-located with the agent.
        for r in self.resources:
            if r.location == agent.location and r.quantity > 0:
                label = "food" if r.drive == DriveKind.HUNGER else (
                    "water" if r.drive == DriveKind.THIRST else "companionship")
                obs.append(Observation(
                    text=f"here there is {label} (quantity {r.quantity})",
                    related_drive=r.drive))
        # Other agents co-located.
        for other in self.agents:
            if other.agent_id != agent.agent_id and other.location == agent.location \
                    and other.drives.alive:
                obs.append(Observation(
                    text=f"{other.agent_id} is here with you",
                    related_drive=DriveKind.AFFILIATION))
        # Adjacent locations (fully connected world for now).
        for loc in self.locations:
            if loc != agent.location:
                obs.append(Observation(text=f"you could move to {loc}",
                                       related_drive=None, base_weight=0.5))
        if not obs:
            obs.append(Observation(text="the area is empty and quiet",
                                   related_drive=None))
        return obs

    # ---- action resolution (proto Game Master) ----

    def resolve_action(self, agent: Agent, action: str) -> str:
        """Apply an action to world state, return a short outcome description."""
        if action == "consume":
            candidates = [r for r in self.resources
                         if r.location == agent.location and r.quantity > 0]
            if not candidates:
                return "tried to consume but found nothing here"
            # When multiple resource types are present, consume whichever
            # matches the agent's currently most urgent drive -- not just
            # whichever happens to be listed first in the resources list. Picking
            # by list order silently starved agents on a co-located second drive
            # (e.g. always eating, never drinking, when both are here).
            r = max(candidates, key=lambda res: agent.drives.states[res.drive].level)
            agent.drives.satisfy(r.drive)
            r.quantity -= 1
            if r.quantity == 0:
                self.logger.log_event(EventRecord(
                    self.logger.run_id, self.tick_count,
                    "resource_depleted", None,
                    f"{r.drive.value} node at {r.location} depleted"))
            return f"consumed {r.drive.value} resource at {agent.location}"

        if action.startswith("move:"):
            dest = action.split(":", 1)[1]
            if dest in self.locations:
                agent.location = dest
                return f"moved to {dest}"
            return f"tried to move to unknown place {dest}"

        if action.startswith("talk:"):
            target = action.split(":", 1)[1]
            agent.drives.satisfy(DriveKind.AFFILIATION)
            return f"talked with {target}"

        if action == "reflect":
            return "paused to reflect"

        return "waited"

    # ---- main loop ----

    def run(self, policy: Callable, ticks: int, reflect_every: int = 5,
            reflect_fn: Callable | None = None) -> None:
        for _ in range(ticks):
            self.tick_count += 1
            for agent in self.agents:
                if not agent.drives.alive:
                    continue

                perceived_obs = self.observations_for(agent)
                biases = agent.drives.salience_biases()
                perceived = render_perceived(perceived_obs, biases, top_k=6)

                action = policy(agent, perceived, self)
                outcome = self.resolve_action(agent, action)
                agent.recent_memories.append(
                    f"tick {self.tick_count}: did '{action}' -> {outcome}")

                # Optional reflection (self-model revision) — LLM-backed in reality.
                if reflect_fn and self.tick_count % reflect_every == 0:
                    reflect_fn(agent, perceived, outcome, self.tick_count)

                # Advance body AFTER acting, then check mortality.
                agent.drives.step(self.tick_count, mortality=self.mortality)
                if not agent.drives.alive and self.mortality:
                    self.logger.log_event(EventRecord(
                        self.logger.run_id, self.tick_count, "death",
                        agent.agent_id,
                        f"died of {agent.drives.cause_of_death}"))

                self.logger.log_tick(TickRecord(
                    run_id=self.logger.run_id,
                    tick=self.tick_count,
                    agent_id=agent.agent_id,
                    alive=agent.drives.alive,
                    drives=agent.drives.snapshot(),
                    position=agent.location,
                    chosen_action=action,
                    perceived=perceived,
                    self_model_version=agent.history.current.version,
                    cause_of_death=agent.drives.cause_of_death,
                ))

            if all(not a.drives.alive for a in self.agents):
                self.logger.log_event(EventRecord(
                    self.logger.run_id, self.tick_count, "extinction", None,
                    "all agents dead"))
                break
