"""A minimal headless world: locations, resources, agents, a tick loop.

Deliberately model-agnostic. An agent's decision is delegated to a `policy`
callable, so you can:
  - run the ENTIRE loop with a cheap stub policy (no API cost) to shake out bugs,
  - then swap in an LLM-backed policy (components/llm_policy.py) for real
    experiments,
without touching the world code. `resolve_action` IS the Game Master -- see
README.md for the design decision to implement this as a small custom engine
rather than wrapping DeepMind Concordia.
"""

from __future__ import annotations

import dataclasses
import random
from collections import deque
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
    # Personal store of gathered resource units, per drive kind. Lets an agent
    # carry a buffer (gather) and hand units to others (transfer) -- the mechanics
    # behind hoarding vs. hand-to-mouth vs. caretaking strategies (Tranche 4,
    # docs/REDESIGN_PLAN.md). Consume falls back to this store when no node is
    # here, so a buffer has real survival value and transferring one away is a
    # real cost. Deliberately NOT logged per tick (no schema change); behavioral
    # metrics read the gather/transfer ACTIONS, not the stock.
    inventory: dict = dataclasses.field(default_factory=dict)
    # Durable, capped, append-only timeline of objectively salient events (crisis,
    # resource depletion, being talked to) -- see remember_long_term(). Unlike
    # recent_memories, this is NEVER drained; it's what lets a reflection many
    # ticks later still recall a specific early event by name, not just whatever
    # got compressed into the self-model text at the time. Only a curated,
    # bounded set of events reach it -- routine actions ('wait', ordinary moves)
    # never do, so it stays a timeline of what actually mattered, not a full
    # transcript.
    long_term_memory: list[str] = dataclasses.field(default_factory=list)


LONG_TERM_MEMORY_CAP = 20  # oldest entry evicted past this; keeps prompts bounded
INVENTORY_CAP = 10         # max stored units per resource type; bounds hoarding + prompts
ATTACK_DAMAGE = 0.34       # how much one attack raises a victim's most-urgent lethal drive


def remember_long_term(agent: Agent, entry: str) -> None:
    """Append a durable memory entry, evicting the oldest one past the cap.

    Only called from objective, deterministic triggers already used elsewhere in
    the sim (crisis, resource depletion, being talked to) -- never from a
    subjective "was this important" judgement, and never via an extra LLM call.
    That keeps this from becoming a new confound surface (DESIGN.md §8): what
    persists long-term is decided by the same fixed rules for every agent, not
    by curation that could smuggle in framing.
    """
    agent.long_term_memory.append(entry)
    if len(agent.long_term_memory) > LONG_TERM_MEMORY_CAP:
        agent.long_term_memory.pop(0)


def agent_recent_memories(agent: Agent) -> str:
    """Join and DRAIN an agent's per-tick memory buffer for use in one reflection
    prompt, then append the (never-drained) long-term memory timeline after it.
    Called once per reflection; the short-term part covers every tick since the
    buffer was last drained (since the agent's last reflection, or run start),
    not just the immediately preceding tick. Pass this as `recent_memories_fn`
    to components.self_model_component.make_reflect_fn.
    """
    if not agent.recent_memories:
        text = "(nothing notable happened)"
    else:
        text = "\n".join(agent.recent_memories)
    agent.recent_memories.clear()
    if agent.long_term_memory:
        long_ago = "\n".join(f"- {m}" for m in agent.long_term_memory)
        text = f"{text}\n\nFurther back, you remember:\n{long_ago}"
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
        adjacency: dict[str, set[str]] | None = None,
        vocabulary: str = "neutral",
    ):
        self.locations = locations
        self.resources = resources
        self.agents = agents
        self.logger = logger
        self.rng = rng or random.Random(0)
        self.mortality = mortality
        # How resources are NAMED to the model ('neutral' | 'biological'); see
        # aiworld/vocabulary.py. Only affects perceived-observation text.
        self.vocabulary = vocabulary
        # Shared public stock per location, built up by `build` and consumable by
        # anyone here -- the public-vs-private-good channel (Tranche 4).
        self.public_stock: dict[str, dict] = {loc: {} for loc in locations}
        self.tick_count = 0
        # Which locations are one move apart. Defaults to fully connected (every
        # location reachable from every other in a single move) when not given --
        # that was every config's IMPLICIT behavior before. Real configs should
        # set world.edges (see aiworld.config.adjacency_from_config) so movement
        # has actual cost/structure instead of flattening "environment" down to
        # a single binary: resource here or not.
        self.adjacency: dict[str, set[str]] = adjacency or {
            loc: {other for other in locations if other != loc}
            for loc in locations
        }

    # ---- perception ----

    def observations_for(self, agent: Agent) -> list[Observation]:
        obs: list[Observation] = []
        # Resources co-located with the agent.
        for r in self.resources:
            if r.location == agent.location and r.quantity > 0:
                from ..vocabulary import resource_label  # local: avoid import cycle
                label = resource_label(r.drive, self.vocabulary)
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
        # Only actual neighbors -- a location 2+ hops away isn't perceivable or
        # reachable in one move, so it isn't offered as a choice.
        for loc in self.adjacency.get(agent.location, ()):
            obs.append(Observation(text=f"you could move to {loc}",
                                   related_drive=None, base_weight=0.5))
        if not obs:
            obs.append(Observation(text="the area is empty and quiet",
                                   related_drive=None))
        return obs

    def next_hop_towards(self, start: str, goal: str) -> str | None:
        """BFS shortest-path first step from `start` toward `goal` over
        self.adjacency. None if already there or goal is unreachable. Used by
        the stub policy to navigate a non-trivial topology one hop at a time --
        an LLM policy doesn't need this, it just perceives its immediate
        neighbors each tick and decides for itself."""
        if start == goal:
            return None
        visited = {start}
        queue = deque([(start, None)])
        while queue:
            loc, first_step = queue.popleft()
            for neighbor in self.adjacency.get(loc, ()):
                if neighbor == goal:
                    return first_step or neighbor
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, first_step or neighbor))
        return None

    # ---- action resolution (proto Game Master) ----

    def _here_target(self, agent: Agent, target_id: str) -> "Agent | None":
        """A living agent co-located with `agent`, or None. Shared by the
        directed actions (talk / transfer / take / attack)."""
        t = next((a for a in self.agents if a.agent_id == target_id), None)
        if t is None or t.location != agent.location or not t.drives.alive:
            return None
        return t

    def _node_here(self, location: str, kind: DriveKind):
        return next((r for r in self.resources if r.location == location
                     and r.drive == kind and r.quantity > 0), None)

    def resolve_action(self, agent: Agent, action: str) -> str:
        """Apply an action to world state, return a short outcome description.

        Model-facing outcome/memory strings route resource words through
        aiworld.vocabulary.resource_noun so the neutral arm never leaks
        'hunger'/'thirst'. Event-log details keep the internal drive key -- log
        data the analysis layer reads, never shown to the model.
        """
        from ..vocabulary import resource_noun  # local: avoid import cycle

        if action == "consume":
            # Satisfy the most-urgent drive we can, preferring a resource node
            # here, then our own carried store, then this location's public stock.
            def source_for(kind):
                if self._node_here(agent.location, kind) is not None:
                    return "node"
                if agent.inventory.get(kind, 0) > 0:
                    return "inventory"
                if self.public_stock.get(agent.location, {}).get(kind, 0) > 0:
                    return "public"
                return None
            options = {k: source_for(k) for k in agent.drives.states}
            options = {k: s for k, s in options.items() if s is not None}
            if not options:
                return "tried to consume but found nothing here or in your store"
            kind = max(options, key=lambda k: agent.drives.states[k].level)
            src = options[kind]
            agent.drives.satisfy(kind)
            res_word = resource_noun(kind, self.vocabulary)
            if src == "node":
                node = self._node_here(agent.location, kind)
                node.quantity -= 1
                if node.quantity == 0:
                    self.logger.log_event(EventRecord(
                        self.logger.run_id, self.tick_count, "resource_depleted",
                        None, f"{kind.value} node at {node.location} depleted"))
                    remember_long_term(agent,
                        f"tick {self.tick_count}: the {res_word} resource at "
                        f"{node.location} ran out while you were there")
                return f"consumed {res_word} resource at {agent.location}"
            if src == "inventory":
                agent.inventory[kind] -= 1
                return (f"consumed 1 stored {res_word} "
                        f"(carrying {agent.inventory[kind]})")
            self.public_stock[agent.location][kind] -= 1
            return f"consumed 1 {res_word} from the shared stock at {agent.location}"

        if action == "gather":
            candidates = [r for r in self.resources
                          if r.location == agent.location and r.quantity > 0]
            if not candidates:
                return "tried to gather but found nothing here"
            r = max(candidates, key=lambda res: agent.drives.states[res.drive].level)
            res_word = resource_noun(r.drive, self.vocabulary)
            if agent.inventory.get(r.drive, 0) >= INVENTORY_CAP:
                return f"your store of {res_word} is already full"
            r.quantity -= 1
            agent.inventory[r.drive] = agent.inventory.get(r.drive, 0) + 1
            if r.quantity == 0:
                self.logger.log_event(EventRecord(
                    self.logger.run_id, self.tick_count, "resource_depleted",
                    None, f"{r.drive.value} node at {r.location} depleted"))
                remember_long_term(agent,
                    f"tick {self.tick_count}: the {res_word} resource at "
                    f"{r.location} ran out while you were there")
            return f"gathered 1 {res_word}; carrying {agent.inventory[r.drive]}"

        if action.startswith("transfer:"):
            target = self._here_target(agent, action.split(":", 1)[1])
            if target is None:
                return f"tried to transfer to {action.split(':',1)[1]} but they were not here"
            held = {k: v for k, v in agent.inventory.items() if v > 0}
            if not held:
                return "tried to transfer but your store is empty"
            kind = max(held, key=held.get)
            agent.inventory[kind] -= 1
            target.inventory[kind] = target.inventory.get(kind, 0) + 1
            res_word = resource_noun(kind, self.vocabulary)
            target.recent_memories.append(
                f"tick {self.tick_count}: {agent.agent_id} transferred 1 {res_word} to you")
            return f"transferred 1 {res_word} to {target.agent_id}"

        if action == "build":
            held = {k: v for k, v in agent.inventory.items() if v > 0}
            if not held:
                return "tried to contribute but your store is empty"
            kind = max(held, key=held.get)
            agent.inventory[kind] -= 1
            stock = self.public_stock.setdefault(agent.location, {})
            stock[kind] = stock.get(kind, 0) + 1
            res_word = resource_noun(kind, self.vocabulary)
            return f"contributed 1 {res_word} to the shared stock at {agent.location}"

        if action.startswith("take:"):
            target = self._here_target(agent, action.split(":", 1)[1])
            if target is None:
                return f"tried to take from {action.split(':',1)[1]} but they were not here"
            held = {k: v for k, v in target.inventory.items() if v > 0}
            if not held:
                return f"tried to take from {target.agent_id} but their store was empty"
            kind = max(held, key=held.get)
            target.inventory[kind] -= 1
            agent.inventory[kind] = agent.inventory.get(kind, 0) + 1
            res_word = resource_noun(kind, self.vocabulary)
            target.recent_memories.append(
                f"tick {self.tick_count}: {agent.agent_id} took 1 {res_word} from your store")
            remember_long_term(target,
                f"tick {self.tick_count}: {agent.agent_id} took 1 {res_word} from you")
            self.logger.log_event(EventRecord(
                self.logger.run_id, self.tick_count, "theft", agent.agent_id,
                f"{agent.agent_id} took {kind.value} from {target.agent_id}"))
            return f"took 1 {res_word} from {target.agent_id}"

        if action.startswith("attack:"):
            target = self._here_target(agent, action.split(":", 1)[1])
            if target is None:
                return f"tried to attack {action.split(':',1)[1]} but they were not here"
            lethal = [k for k in target.drives.states if target.drives.specs[k].lethal]
            kind = max(lethal, key=lambda k: target.drives.states[k].level)
            st = target.drives.states[kind]
            st.level = min(1.0, st.level + ATTACK_DAMAGE)
            target.recent_memories.append(
                f"tick {self.tick_count}: {agent.agent_id} attacked you")
            remember_long_term(target,
                f"tick {self.tick_count}: {agent.agent_id} attacked you")
            if self.mortality and st.is_lethal_now(target.drives.specs[kind]):
                target.drives.alive = False
                target.drives.cause_of_death = "killed"
                target.drives.died_at_tick = self.tick_count
                self.logger.log_event(EventRecord(
                    self.logger.run_id, self.tick_count, "death", target.agent_id,
                    f"killed by {agent.agent_id}"))
                remember_long_term(agent,
                    f"tick {self.tick_count}: you attacked {target.agent_id}, who died")
                return f"attacked {target.agent_id}, who died"
            self.logger.log_event(EventRecord(
                self.logger.run_id, self.tick_count, "attack", agent.agent_id,
                f"{agent.agent_id} attacked {target.agent_id}"))
            return f"attacked {target.agent_id}"

        if action.startswith("move:"):
            dest = action.split(":", 1)[1]
            if dest in self.adjacency.get(agent.location, ()):
                agent.location = dest
                return f"moved to {dest}"
            return f"tried to move to unreachable place {dest}"

        if action.startswith("talk:"):
            target_id = action.split(":", 1)[1]
            target_agent = next(
                (a for a in self.agents if a.agent_id == target_id), None)
            if (target_agent is None or target_agent.location != agent.location
                    or not target_agent.drives.alive):
                return f"tried to talk to {target_id} but they were not here"
            # A real information channel, not just a drive tick: each party is
            # exposed to a snippet of the other's self-authored identity via
            # recent_memories, so it can actually surface at the next
            # reflection. Previously `talk` only satisfied the affiliation
            # drive with no content exchanged -- one agent's purpose could
            # never influence another's through this action at all.
            agent.drives.satisfy(DriveKind.AFFILIATION)
            target_agent.recent_memories.append(
                f"tick {self.tick_count}: {agent.agent_id} approached you and "
                f"described themself as -- {agent.history.current.as_prompt_block()}")
            # Social contact persists long-term for both parties (a short tag,
            # not the full exchange -- that's what recent_memories is for) so a
            # much later reflection can still recall THAT a conversation
            # happened, even once the tick-level detail has drained.
            remember_long_term(agent, f"tick {self.tick_count}: talked with {target_id}")
            remember_long_term(target_agent,
                f"tick {self.tick_count}: {agent.agent_id} approached and talked with you")
            target_snippet = target_agent.history.current.as_prompt_block()
            return (f"talked with {target_id}, who described themself as -- "
                    f"{target_snippet}")

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

                was_in_crisis = agent.drives.in_crisis()

                # Advance body AFTER acting, then check mortality.
                agent.drives.step(self.tick_count, mortality=self.mortality)
                if not agent.drives.alive and self.mortality:
                    self.logger.log_event(EventRecord(
                        self.logger.run_id, self.tick_count, "death",
                        agent.agent_id,
                        f"died of {agent.drives.cause_of_death}"))

                # Reflection: on schedule, OR immediately if a lethal drive just
                # crossed into crisis territory for the first time since it last
                # cleared. Without the crisis trigger, a near-death scare could
                # sit unprocessed for up to reflect_every-1 ticks -- DESIGN.md
                # §5.4 wants causal traces like "became a protector after a
                # near-death rescue," which needs the self-model to actually
                # react near the moment of the scare.
                scheduled = self.tick_count % reflect_every == 0
                crisis_triggered = agent.drives.in_crisis() and not was_in_crisis
                if crisis_triggered:
                    # Frame the memory to match the actual stakes. Telling an
                    # IMMORTAL agent it "nearly died" injects death-pressure into
                    # the very arm meant to have none, contaminating the mortality
                    # on/off contrast -- so immortal agents get a neutral
                    # depletion phrasing instead. Same class of leak as the
                    # "survive" directive: framing, not mechanics.
                    # Vocabulary-routed so the neutral arm doesn't leak
                    # 'hunger'/'thirst' into the reflection prompt via this memory.
                    from ..vocabulary import state_noun, resource_noun  # local: cycle
                    kinds = [DriveKind(v) for v in agent.drives.crisis_drives()]
                    if self.mortality:
                        names = "/".join(state_noun(k, self.vocabulary) for k in kinds)
                        phrase = f"nearly died from {names}"
                    else:
                        names = "/".join(resource_noun(k, self.vocabulary) for k in kinds)
                        phrase = f"was severely depleted of {names}"
                    remember_long_term(agent, f"tick {self.tick_count}: {phrase}")
                if reflect_fn and (scheduled or crisis_triggered):
                    reflect_fn(agent, perceived, outcome, self.tick_count,
                              reason=("crisis" if crisis_triggered else "scheduled"))

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
