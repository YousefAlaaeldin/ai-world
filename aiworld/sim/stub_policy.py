"""A cheap, deterministic-ish stub policy — NO LLM.

Purpose: exercise the full tick loop, mortality, resource depletion, and logging
without any API cost. It is intentionally dumb: it does NOT form purpose, it just
keeps agents alive-ish so we can verify the mechanics and the analysis pipeline on
real output files.

The real experiment replaces this with an LLM-backed policy (see
components/llm_policy.py, phase 1). Do not read anything into stub behavior — it is
scaffolding, not a result.
"""

from __future__ import annotations

import random

from ..drives import DriveKind
from .world import Agent, World


def make_stub_policy(rng: random.Random | None = None):
    rng = rng or random.Random(0)

    def policy(agent: Agent, perceived: str, world: World) -> str:
        # Find the most urgent lethal drive.
        hunger = agent.drives.states[DriveKind.HUNGER].level
        thirst = agent.drives.states[DriveKind.THIRST].level
        urgent = DriveKind.THIRST if thirst >= hunger else DriveKind.HUNGER

        here = [r for r in world.resources
                if r.location == agent.location and r.quantity > 0]
        # If the needed resource is here, consume it.
        if any(r.drive == urgent for r in here) and max(hunger, thirst) > 0.3:
            return "consume"
        # Otherwise head toward some location that has the needed resource --
        # one hop at a time (world.next_hop_towards), since the world isn't
        # fully connected: a resource 2+ locations away can't be jumped to
        # directly, only navigated toward.
        targets = [r.location for r in world.resources
                   if r.drive == urgent and r.quantity > 0
                   and r.location != agent.location]
        if targets and max(hunger, thirst) > 0.2:
            goal = rng.choice(targets)
            next_hop = world.next_hop_towards(agent.location, goal)
            if next_hop:
                return f"move:{next_hop}"
        # Low pressure: sometimes socialize, sometimes wander, sometimes wait.
        roll = rng.random()
        if roll < 0.2:
            others = [a.agent_id for a in world.agents
                      if a.agent_id != agent.agent_id and a.location == agent.location
                      and a.drives.alive]
            if others:
                return f"talk:{rng.choice(others)}"
        if roll < 0.5:
            neighbors = list(world.adjacency.get(agent.location, ()))
            if neighbors:
                return f"move:{rng.choice(neighbors)}"
        return "wait"

    return policy
