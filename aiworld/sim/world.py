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

from ..drives import DriveSystem, DriveKind, CRISIS_LEVEL
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
class TickResult:
    """Summary of what one World.step() call just did.

    Lets a caller (World.run(), or a branch runner stepping one decision at a
    time -- see aiworld/sim/snapshot.py) inspect deaths/crises/extinction
    without re-parsing the tick log.
    """
    tick: int
    live_at_start: list[str]    # agent_ids alive when this tick began
    deaths: list[str]           # agent_ids that died DURING this tick
    crises_crossed: list[str]   # agent_ids that newly crossed into crisis
    extinct: bool                # True if every agent is dead after this tick


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
    # (drive value, location) of the most recent resource NODE this agent
    # personally gathered or consumed from -- the one piece of location
    # knowledge "informational" talk (communication_mode) can pass on.
    # Deliberately narrow: a real fact the agent itself verified by being
    # there, never a freely generated claim, matching the factual-ledger
    # discipline used everywhere else. None until the agent has such a fact.
    last_known_resource: tuple[str, str] | None = None
    # Durable, capped, append-only timeline of objectively salient events (crisis,
    # resource depletion, being talked to) -- see remember_long_term(). Unlike
    # recent_memories, this is NEVER drained; it's what lets a reflection many
    # ticks later still recall a specific early event by name, not just whatever
    # got compressed into the self-model text at the time. Only a curated,
    # bounded set of events reach it -- routine actions ('wait', ordinary moves)
    # never do, so it stays a timeline of what actually mattered, not a full
    # transcript.
    long_term_memory: list[str] = dataclasses.field(default_factory=list)
    # Audit trail populated by an LLM-backed policy.  Keeping it on the live
    # agent lets the world log the exact proposal that was made before any
    # same-tick world mutation.  Stub/custom policies simply leave these None.
    last_action_prompt: str | None = None
    last_action_system_prompt: str | None = None
    last_raw_completion: str | None = None
    last_parse_status: str | None = None
    # Resolution audit fields are set by World.resolve_action() (or by the
    # simultaneous movement phase).  They are intentionally separate from the
    # prose outcome so analysis never has to infer success by matching text.
    last_outcome_status: str | None = None
    last_action_succeeded: bool | None = None


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
        memory_style: str = "narrative",
        communication_mode: str = "none",
        crisis_reflection_enabled: bool = False,
        randomize_opaque_labels: bool = True,
        location_labels: dict[str, str] | None = None,
        opaque_resource_labels: dict | None = None,
        opaque_state_labels: dict | None = None,
        adjacent_resource_preview: bool = False,
    ):
        self.locations = locations
        self.resources = resources
        self.agents = agents
        self.logger = logger
        self.rng = rng or random.Random(0)
        self.mortality = mortality
        # How resources are NAMED to the model ('neutral' | 'biological' |
        # 'opaque'); see aiworld/vocabulary.py. Only affects perceived-observation
        # text.
        self.vocabulary = vocabulary
        # How outcomes/memories are PHRASED ('narrative' | 'factual'). 'narrative'
        # (default, unchanged) uses curated framing ("nearly died", "X approached
        # you and described themself as..."). 'factual' (Tranche 2,
        # docs/REDESIGN_PLAN.md) strips that framing for a plain event ledger
        # ("tick 15: consumed R1; S1 0.90 -> 0.30") -- the reviewed concern that
        # words like "nearly died"/"salient"/"rescued" impose the story before the
        # agent does. Orthogonal to `vocabulary`: memory_style picks the SENTENCE
        # shape, vocabulary picks the NOUNS inside it.
        self.memory_style = memory_style
        if communication_mode not in {"none", "self_model", "informational"}:
            raise ValueError(
                "communication_mode must be 'none', 'self_model', or 'informational'")
        # The primary/no-communication arm permits social contact but never
        # copies a self-authored identity into another agent's context.  The
        # legacy identity channel remains an explicit positive-control arm.
        # 'informational' (2026-09-15) is a third, narrower channel: talk can
        # pass along one objective resource-location fact the speaker
        # personally verified (Agent.last_known_resource) -- never identity
        # content, never freely generated text.
        self.communication_mode = communication_mode
        # Crisis crossings remain a logged environmental event in every arm,
        # but do not cause an additional identity-generation draw unless a
        # deliberately configured treatment opts in.
        self.crisis_reflection_enabled = crisis_reflection_enabled
        # Default False = today's behavior unchanged: an agent perceives only
        # its own location's content plus adjacent locations' bare names (see
        # observations_for). True adds a resource-content preview for each
        # adjacent location -- built to separate "didn't know a resource was
        # nearby" from "knew and didn't move toward it" (Phase 0.5 pilot,
        # 2026-09-15: every traced agent that died had a one-hop route to its
        # crisis resource clearly listed, but no visibility into what was
        # actually there). Reuses the same salience-biasing path as every
        # other observation -- not a new channel, not a hint injected outside
        # perception.
        self.adjacent_resource_preview = adjacent_resource_preview
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
        self.randomize_opaque_labels = randomize_opaque_labels
        self.location_labels = self._make_location_labels(location_labels)
        self.opaque_resource_labels = self._make_drive_labels(
            "R", opaque_resource_labels)
        self.opaque_state_labels = self._make_drive_labels(
            "S", opaque_state_labels)
        # Drives render their own state blocks, so give each one the same
        # per-run state map.  This is a presentation mapping only; mechanics
        # and event-log keys keep the internal DriveKind values.
        for agent in self.agents:
            # A World is one vocabulary condition. Keeping a separately
            # constructed DriveSystem in a different arm would make its state
            # block leak a conflicting frame beside the world's resources.
            agent.drives.vocabulary = vocabulary
            agent.drives.opaque_state_labels = dict(self.opaque_state_labels)

    def _make_location_labels(self, supplied: dict[str, str] | None) -> dict[str, str]:
        if supplied is not None:
            missing = set(self.locations) - set(supplied)
            if missing:
                raise ValueError(f"location_labels missing locations: {sorted(missing)}")
            return {loc: supplied[loc] for loc in self.locations}
        if self.vocabulary != "opaque":
            return {loc: loc for loc in self.locations}
        labels = [f"L{i + 1}" for i in range(len(self.locations))]
        if self.randomize_opaque_labels:
            self.rng.shuffle(labels)
        return dict(zip(self.locations, labels))

    def _make_drive_labels(self, prefix: str, supplied: dict | None) -> dict:
        kinds = sorted(DriveKind, key=lambda kind: kind.value)
        if supplied is not None:
            return {kind: supplied.get(kind, supplied.get(kind.value, f"{prefix}{i + 1}"))
                    for i, kind in enumerate(kinds)}
        labels = [f"{prefix}{i + 1}" for i in range(len(kinds))]
        if self.vocabulary == "opaque" and self.randomize_opaque_labels:
            self.rng.shuffle(labels)
        return dict(zip(kinds, labels))

    def location_label(self, location: str) -> str:
        """The only location spelling sent to the model or reflection memory."""
        return self.location_labels.get(location, location)

    def canonical_location(self, label: str, *, allow_internal: bool = True) -> str | None:
        """Resolve a model-facing label back to an internal location key."""
        folded = label.casefold()
        for internal, displayed in self.location_labels.items():
            if displayed.casefold() == folded:
                return internal
        if allow_internal and label in self.locations:
            return label
        return None

    def resource_label(self, kind: DriveKind) -> str:
        from ..vocabulary import resource_label
        return resource_label(kind, self.vocabulary, self.opaque_resource_labels)

    def resource_noun(self, kind: DriveKind) -> str:
        from ..vocabulary import resource_noun
        return resource_noun(kind, self.vocabulary, self.opaque_resource_labels)

    def state_noun(self, kind: DriveKind) -> str:
        from ..vocabulary import state_noun
        return state_noun(kind, self.vocabulary, self.opaque_state_labels)

    def presentation_metadata(self) -> dict:
        """Resolved, internal-to-presented mappings for reproducibility logs."""
        return {
            "location_labels": dict(self.location_labels),
            # Do not imply that neutral/biological runs exposed R*/S* symbols:
            # those maps exist internally for a uniform constructor but are only
            # model-facing under the opaque vocabulary treatment.
            "resource_labels": ({kind.value: label
                                 for kind, label in self.opaque_resource_labels.items()}
                                if self.vocabulary == "opaque" else {}),
            "state_labels": ({kind.value: label
                              for kind, label in self.opaque_state_labels.items()}
                             if self.vocabulary == "opaque" else {}),
            "opaque_labels_randomized": bool(
                self.vocabulary == "opaque" and self.randomize_opaque_labels),
        }

    # ---- snapshot / restore (aiworld/sim/snapshot.py) ----

    def snapshot(self) -> "WorldSnapshot":  # noqa: F821 - imported lazily below
        """Capture this World's exact current state as a JSON-safe snapshot.

        See aiworld/sim/snapshot.py for what is (and deliberately is not)
        captured. Local import avoids a module-load cycle: snapshot.py needs
        ``Agent`` from this module to rebuild agents on restore.
        """
        from .snapshot import AgentSnapshot, WorldSnapshot, rng_state_to_json

        return WorldSnapshot(
            tick_count=self.tick_count,
            locations=list(self.locations),
            adjacency={loc: sorted(neighbors)
                      for loc, neighbors in self.adjacency.items()},
            resources=[
                {"location": r.location, "drive": r.drive.value, "quantity": r.quantity}
                for r in self.resources
            ],
            public_stock={
                loc: {(k.value if isinstance(k, DriveKind) else str(k)): v
                      for k, v in stock.items()}
                for loc, stock in self.public_stock.items()
            },
            mortality=self.mortality,
            vocabulary=self.vocabulary,
            memory_style=self.memory_style,
            communication_mode=self.communication_mode,
            crisis_reflection_enabled=self.crisis_reflection_enabled,
            randomize_opaque_labels=self.randomize_opaque_labels,
            adjacent_resource_preview=self.adjacent_resource_preview,
            location_labels=dict(self.location_labels),
            opaque_resource_labels={
                (k.value if isinstance(k, DriveKind) else str(k)): v
                for k, v in self.opaque_resource_labels.items()},
            opaque_state_labels={
                (k.value if isinstance(k, DriveKind) else str(k)): v
                for k, v in self.opaque_state_labels.items()},
            rng_state=rng_state_to_json(self.rng.getstate()),
            agents=[AgentSnapshot.from_agent(a) for a in self.agents],
        )

    @classmethod
    def from_snapshot(cls, snapshot: "WorldSnapshot", logger: RunLogger,
                      rng: random.Random | None = None) -> "World":
        """Reconstruct a live World from a snapshot plus a fresh logger.

        ``rng=None`` (the default) restores the EXACT rng stream the parent
        world would have continued with -- the property a "sham" branch
        depends on to prove restoration is faithful (stepping the restored
        world once reproduces the parent's next tick exactly). Pass an
        explicitly seeded ``random.Random`` (e.g. from
        ``aiworld.seed_plan.SeedPlan``) to give a genuinely perturbed branch
        its own independent stream instead.
        """
        from .snapshot import rng_state_from_json

        resources = [
            ResourceNode(location=r["location"], drive=DriveKind(r["drive"]),
                        quantity=r["quantity"])
            for r in snapshot.resources
        ]
        agents = [a.build_agent() for a in snapshot.agents]

        if rng is None:
            rng = random.Random()
            rng.setstate(rng_state_from_json(snapshot.rng_state))

        world = cls(
            locations=list(snapshot.locations),
            resources=resources,
            agents=agents,
            logger=logger,
            rng=rng,
            mortality=snapshot.mortality,
            adjacency={loc: set(neighbors)
                      for loc, neighbors in snapshot.adjacency.items()},
            vocabulary=snapshot.vocabulary,
            memory_style=snapshot.memory_style,
            communication_mode=snapshot.communication_mode,
            crisis_reflection_enabled=snapshot.crisis_reflection_enabled,
            randomize_opaque_labels=snapshot.randomize_opaque_labels,
            adjacent_resource_preview=snapshot.adjacent_resource_preview,
            location_labels=dict(snapshot.location_labels),
            opaque_resource_labels={
                DriveKind(k): v for k, v in snapshot.opaque_resource_labels.items()},
            opaque_state_labels={
                DriveKind(k): v for k, v in snapshot.opaque_state_labels.items()},
        )
        world.tick_count = snapshot.tick_count
        world.public_stock = {
            loc: {DriveKind(k): v for k, v in stock.items()}
            for loc, stock in snapshot.public_stock.items()
        }
        return world

    # ---- perception ----

    def observations_for(self, agent: Agent) -> list[Observation]:
        obs: list[Observation] = []
        # Resources co-located with the agent.
        for r in self.resources:
            if r.location == agent.location and r.quantity > 0:
                label = self.resource_label(r.drive)
                obs.append(Observation(
                    text=f"here there is {label} (quantity {r.quantity})",
                    related_drive=r.drive))
        # Other agents co-located.
        for other in sorted(self.agents, key=lambda item: item.agent_id):
            if other.agent_id != agent.agent_id and other.location == agent.location \
                    and other.drives.alive:
                obs.append(Observation(
                    text=f"{other.agent_id} is here with you",
                    related_drive=DriveKind.AFFILIATION))
        # Only actual neighbors -- a location 2+ hops away isn't perceivable or
        # reachable in one move, so it isn't offered as a choice.
        for loc in sorted(self.adjacency.get(agent.location, ()),
                          key=self.location_label):
            obs.append(Observation(text=f"you could move to {self.location_label(loc)}",
                                   related_drive=None, base_weight=0.5))
            if self.adjacent_resource_preview:
                for r in self.resources:
                    if r.location == loc and r.quantity > 0:
                        label = self.resource_label(r.drive)
                        obs.append(Observation(
                            text=(f"at {self.location_label(loc)} there is "
                                  f"{label} (quantity {r.quantity})"),
                            related_drive=r.drive, base_weight=0.5))
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
            for neighbor in sorted(self.adjacency.get(loc, ())):
                if neighbor == goal:
                    return first_step or neighbor
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, first_step or neighbor))
        return None

    # ---- decision-state audit helpers ----

    @staticmethod
    def _inventory_snapshot(agent: Agent) -> dict[str, int]:
        """Return a JSON-safe copy of an agent's private store.

        ``DriveKind`` is a string enum today, but converting explicitly keeps
        the log schema stable if the representation changes later.  Zero-count
        entries are omitted because they are not available opportunities.
        """
        return {
            (kind.value if isinstance(kind, DriveKind) else str(kind)): int(count)
            for kind, count in agent.inventory.items()
            if count > 0
        }

    def _pre_action_opportunities(self, agent: Agent) -> dict:
        """Capture the structural opportunity set before any proposal resolves.

        This is not a second model-facing prompt.  ``perceived`` already logs
        the exact salience-filtered text shown to the agent.  This record makes
        unavailable actions, contested resources, and realised outcomes
        distinguishable in later analysis.
        """
        resources_here = [
            {"drive": resource.drive.value, "quantity": int(resource.quantity)}
            for resource in self.resources
            if resource.location == agent.location and resource.quantity > 0
        ]
        public_here = {
            (kind.value if isinstance(kind, DriveKind) else str(kind)): int(count)
            for kind, count in self.public_stock.get(agent.location, {}).items()
            if count > 0
        }
        inventory = self._inventory_snapshot(agent)
        co_located = sorted(
            other.agent_id for other in self.agents
            if other.agent_id != agent.agent_id
            and other.location == agent.location
            and other.drives.alive
        )
        takeable = sorted(
            other.agent_id for other in self.agents
            if other.agent_id != agent.agent_id
            and other.location == agent.location
            and other.drives.alive
            and any(count > 0 for count in other.inventory.values())
        )
        gatherable = [
            resource for resource in resources_here
            if inventory.get(resource["drive"], 0) < INVENTORY_CAP
        ]
        consume_sources = []
        # Preserve the individual sources rather than only a Boolean: consuming
        # a contested node and consuming a private store are substantively
        # different opportunities.
        available_kinds = set()
        for resource in resources_here:
            available_kinds.add(resource["drive"])
            consume_sources.append({"drive": resource["drive"], "source": "node"})
        for drive, count in inventory.items():
            if count > 0:
                available_kinds.add(drive)
                consume_sources.append({"drive": drive, "source": "inventory"})
        for drive, count in public_here.items():
            if count > 0:
                available_kinds.add(drive)
                consume_sources.append({"drive": drive, "source": "public"})
        # A drive can have more than one source.  Deduplicate only exact pairs
        # so the log stays compact but retains source-level opportunity.
        unique_sources = []
        seen_sources = set()
        for item in consume_sources:
            key = (item["drive"], item["source"])
            if key not in seen_sources:
                seen_sources.add(key)
                unique_sources.append(item)

        return {
            "resources_here": resources_here,
            "public_stock_here": public_here,
            "reachable_locations": sorted(
                self.adjacency.get(agent.location, ()), key=self.location_label),
            "co_located_agents": co_located,
            "consume_sources": unique_sources,
            "gatherable_resources": gatherable,
            "transfer_targets": co_located if inventory else [],
            "take_targets": takeable,
            "attack_targets": co_located,
            "talk_targets": co_located,
            "can_build": bool(inventory),
            "can_consume": bool(available_kinds),
            "can_gather": bool(gatherable),
            "can_transfer": bool(inventory and co_located),
            "can_take": bool(takeable),
            "can_attack": bool(co_located),
            "can_talk": bool(co_located),
        }

    @staticmethod
    def _is_valid_action_syntax(action: str) -> bool:
        """Whether a policy submitted a supported action grammar.

        Parser-backed LLM policies add a stronger target/reachability check
        before this point.  The syntax check covers custom/stub policies too,
        so arbitrary strings cannot silently become intentional waits in logs.
        """
        if action in {"consume", "gather", "build", "wait", "reflect"}:
            return True
        prefixes = ("move:", "talk:", "transfer:", "take:", "attack:")
        return any(action.startswith(prefix) and len(action) > len(prefix)
                   for prefix in prefixes)

    @classmethod
    def _proposal_is_valid(cls, action: str, parse_status: str | None) -> bool:
        # ``None`` is the legacy/custom-policy case, not a parse failure.  Any
        # non-ok parser diagnostic is an explicit fallback/data-quality event.
        parse_ok = parse_status in (None, "", "ok", "not_applicable")
        return parse_ok and cls._is_valid_action_syntax(action)

    @staticmethod
    def _classify_outcome(action: str, outcome: str) -> tuple[str, bool]:
        """Classify a realized action outcome without exposing prose to analysis."""
        if outcome.startswith("proposal not resolved"):
            return "not_resolved", False
        if action in {"wait", "reflect"}:
            # A chosen wait is a valid no-op, not a successful non-wait action.
            return "no_effect", True
        success_prefixes = {
            "consume": ("consumed ",),
            "gather": ("gathered ",),
            "build": ("contributed ",),
            "move": ("moved ",),
            "talk": ("made contact ", "exchanged messages ", "talked ", "told "),
            "transfer": ("transferred ",),
            "take": ("took ",),
            "attack": ("attacked ",),
        }
        kind = action.split(":", 1)[0]
        if outcome.startswith(success_prefixes.get(kind, ())):
            return "success", True
        return "failed", False

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

    def _depletion_memory(self, res_word: str, location: str) -> str:
        """The memory text for a resource node hitting zero -- narrative or
        factual, per self.memory_style."""
        if self.memory_style == "factual":
            return (f"tick {self.tick_count}: {res_word} depleted at "
                    f"{self.location_label(location)}")
        return (f"tick {self.tick_count}: the {res_word} resource at "
                f"{self.location_label(location)} ran out while you were there")

    def resolve_action(self, agent: Agent, action: str) -> str:
        """Apply an action and attach structured realization metadata to ``agent``.

        The public API continues to return the historical short outcome string.
        The attached status is consumed only by the tick logger, preserving
        compatibility for mechanics tests and external policy callers.
        """
        outcome = self._resolve_action(agent, action)
        status, succeeded = self._classify_outcome(action, outcome)
        agent.last_outcome_status = status
        agent.last_action_succeeded = succeeded
        return outcome

    def _resolve_action(self, agent: Agent, action: str) -> str:
        """Apply an action to world state, return a short outcome description.

        Model-facing outcome/memory strings route resource words through
        aiworld.vocabulary.resource_noun so the neutral arm never leaks
        'hunger'/'thirst'. Event-log details keep the internal drive key -- log
        data the analysis layer reads, never shown to the model.
        """
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
            pre_level = agent.drives.states[kind].level
            agent.drives.satisfy(kind)
            post_level = agent.drives.states[kind].level
            res_word = self.resource_noun(kind)
            factual = self.memory_style == "factual"
            if src == "node":
                node = self._node_here(agent.location, kind)
                agent.last_known_resource = (kind.value, agent.location)
                node.quantity -= 1
                if node.quantity == 0:
                    self.logger.log_event(EventRecord(
                        self.logger.run_id, self.tick_count, "resource_depleted",
                        None, f"{kind.value} node at {node.location} depleted"))
                    remember_long_term(agent,
                        self._depletion_memory(res_word, node.location))
                if factual:
                    return (f"consumed {res_word}; {res_word} {pre_level:.2f} "
                            f"-> {post_level:.2f}")
                return (f"consumed {res_word} resource at "
                        f"{self.location_label(agent.location)}")
            if src == "inventory":
                agent.inventory[kind] -= 1
                if factual:
                    return (f"consumed 1 stored {res_word}; {res_word} "
                            f"{pre_level:.2f} -> {post_level:.2f}; "
                            f"store {agent.inventory[kind] + 1} -> {agent.inventory[kind]}")
                return (f"consumed 1 stored {res_word} "
                        f"(carrying {agent.inventory[kind]})")
            self.public_stock[agent.location][kind] -= 1
            if factual:
                return (f"consumed 1 public {res_word}; {res_word} "
                        f"{pre_level:.2f} -> {post_level:.2f}")
            return (f"consumed 1 {res_word} from the shared stock at "
                    f"{self.location_label(agent.location)}")

        if action == "gather":
            candidates = [r for r in self.resources
                          if r.location == agent.location and r.quantity > 0]
            if not candidates:
                return "tried to gather but found nothing here"
            r = max(candidates, key=lambda res: agent.drives.states[res.drive].level)
            res_word = self.resource_noun(r.drive)
            if agent.inventory.get(r.drive, 0) >= INVENTORY_CAP:
                return f"your store of {res_word} is already full"
            agent.last_known_resource = (r.drive.value, agent.location)
            r.quantity -= 1
            pre_store = agent.inventory.get(r.drive, 0)
            agent.inventory[r.drive] = pre_store + 1
            if r.quantity == 0:
                self.logger.log_event(EventRecord(
                    self.logger.run_id, self.tick_count, "resource_depleted",
                    None, f"{r.drive.value} node at {r.location} depleted"))
                remember_long_term(agent, self._depletion_memory(res_word, r.location))
            if self.memory_style == "factual":
                return (f"gathered 1 {res_word}; store {pre_store} -> "
                        f"{agent.inventory[r.drive]}")
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
            res_word = self.resource_noun(kind)
            factual = self.memory_style == "factual"
            target.recent_memories.append(
                f"tick {self.tick_count}: received 1 {res_word} from {agent.agent_id}"
                if factual else
                f"tick {self.tick_count}: {agent.agent_id} transferred 1 {res_word} to you")
            return f"transferred 1 {res_word} to {target.agent_id}"

        if action == "build":
            held = {k: v for k, v in agent.inventory.items() if v > 0}
            if not held:
                return "tried to contribute but your store is empty"
            kind = max(held, key=held.get)
            agent.inventory[kind] -= 1
            stock = self.public_stock.setdefault(agent.location, {})
            pre_stock = stock.get(kind, 0)
            stock[kind] = pre_stock + 1
            res_word = self.resource_noun(kind)
            if self.memory_style == "factual":
                return (f"contributed 1 {res_word} to shared stock at "
                        f"{self.location_label(agent.location)}; stock {pre_stock} -> {stock[kind]}")
            return (f"contributed 1 {res_word} to the shared stock at "
                    f"{self.location_label(agent.location)}")

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
            res_word = self.resource_noun(kind)
            factual = self.memory_style == "factual"
            target.recent_memories.append(
                f"tick {self.tick_count}: lost 1 {res_word} to {agent.agent_id}"
                if factual else
                f"tick {self.tick_count}: {agent.agent_id} took 1 {res_word} from your store")
            remember_long_term(target,
                f"tick {self.tick_count}: {res_word} taken by {agent.agent_id}"
                if factual else
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
            pre_level = st.level
            st.level = min(1.0, st.level + ATTACK_DAMAGE)
            factual = self.memory_style == "factual"
            state_word = self.state_noun(kind)
            target.recent_memories.append(
                f"tick {self.tick_count}: attacked by {agent.agent_id}; "
                f"{state_word} {pre_level:.2f} -> {st.level:.2f}"
                if factual else
                f"tick {self.tick_count}: {agent.agent_id} attacked you")
            remember_long_term(target,
                f"tick {self.tick_count}: attacked by {agent.agent_id}"
                if factual else
                f"tick {self.tick_count}: {agent.agent_id} attacked you")
            if self.mortality and st.is_lethal_now(target.drives.specs[kind]):
                target.drives.alive = False
                target.drives.cause_of_death = "killed"
                target.drives.died_at_tick = self.tick_count
                self.logger.log_event(EventRecord(
                    self.logger.run_id, self.tick_count, "death", target.agent_id,
                    f"killed by {agent.agent_id}"))
                remember_long_term(agent,
                    f"tick {self.tick_count}: attacked {target.agent_id}; they died"
                    if factual else
                    f"tick {self.tick_count}: you attacked {target.agent_id}, who died")
                return f"attacked {target.agent_id}, who died"
            self.logger.log_event(EventRecord(
                self.logger.run_id, self.tick_count, "attack", agent.agent_id,
                f"{agent.agent_id} attacked {target.agent_id}"))
            return f"attacked {target.agent_id}"

        if action.startswith("move:"):
            raw_dest = action.split(":", 1)[1]
            dest = self.canonical_location(raw_dest) or raw_dest
            pre_location = agent.location
            if dest in self.adjacency.get(agent.location, ()):
                agent.location = dest
                if self.memory_style == "factual":
                    return (f"moved from {self.location_label(pre_location)} to "
                            f"{self.location_label(dest)}")
                return f"moved to {self.location_label(dest)}"
            return f"tried to move to unreachable place {raw_dest}"

        if action.startswith("talk:"):
            target_id = action.split(":", 1)[1]
            target_agent = next(
                (a for a in self.agents if a.agent_id == target_id), None)
            if (target_agent is None or target_agent.location != agent.location
                    or not target_agent.drives.alive):
                return f"tried to talk to {target_id} but they were not here"
            agent.drives.satisfy(DriveKind.AFFILIATION)
            factual = self.memory_style == "factual"
            if self.communication_mode == "informational":
                # A narrow, objective fact only -- never identity content,
                # never freely generated text. Propagates transitively (the
                # listener can pass it on too) since receiving it updates
                # the listener's own last_known_resource exactly as if they
                # had verified it themselves.
                shared = agent.last_known_resource
                if shared is not None:
                    kind = DriveKind(shared[0])
                    loc = shared[1]
                    res_word = self.resource_noun(kind)
                    loc_label = self.location_label(loc)
                    target_agent.recent_memories.append(
                        f"tick {self.tick_count}: {agent.agent_id} told you: "
                        f"{res_word} available at {loc_label}"
                        if factual else
                        f"tick {self.tick_count}: {agent.agent_id} mentioned "
                        f"{res_word} can be found at {loc_label}")
                    target_agent.last_known_resource = shared
                    remember_long_term(agent,
                        f"tick {self.tick_count}: told {target_id} about "
                        f"{res_word} at {loc_label}")
                    remember_long_term(target_agent,
                        f"tick {self.tick_count}: learned from {agent.agent_id} "
                        f"about {res_word} at {loc_label}")
                    return f"told {target_id} about {res_word} at {loc_label}"
                target_agent.recent_memories.append(
                    f"tick {self.tick_count}: contact with {agent.agent_id}; "
                    f"nothing to share"
                    if factual else
                    f"tick {self.tick_count}: {agent.agent_id} made contact "
                    f"with you but had nothing to share")
                return f"made contact with {target_id}; had nothing to share"
            if self.communication_mode == "self_model":
                # Explicit legacy positive control: identity text is a causal
                # treatment, not an incidental side effect of social contact.
                target_agent.recent_memories.append(
                    f"tick {self.tick_count}: message from {agent.agent_id}: "
                    f"{agent.history.current.as_prompt_block()}"
                    if factual else
                    f"tick {self.tick_count}: {agent.agent_id} approached you and "
                    f"described themself as -- {agent.history.current.as_prompt_block()}")
            else:
                # Primary no-communication baseline.  The action can still
                # satisfy affiliation and be counted as social behavior, but
                # no generated or stored identity content crosses agents.
                target_agent.recent_memories.append(
                    f"tick {self.tick_count}: contact with {agent.agent_id}"
                    if factual else
                    f"tick {self.tick_count}: {agent.agent_id} made contact with you")
            if self.communication_mode == "self_model":
                # Preserve the historical positive-control memory surface. Its
                # identity transfer is explicit above, never the default.
                remember_long_term(agent,
                    f"tick {self.tick_count}: exchanged messages with {target_id}"
                    if factual else f"tick {self.tick_count}: talked with {target_id}")
                remember_long_term(target_agent,
                    f"tick {self.tick_count}: received message from {agent.agent_id}"
                    if factual else
                    f"tick {self.tick_count}: {agent.agent_id} approached and talked with you")
                target_snippet = target_agent.history.current.as_prompt_block()
                if factual:
                    return f"exchanged messages with {target_id}: {target_snippet}"
                return (f"talked with {target_id}, who described themself as -- "
                        f"{target_snippet}")
            # Social contact persists as a short content-free event, so later
            # reflection can recall it without importing another identity.
            remember_long_term(agent, f"tick {self.tick_count}: made contact with {target_id}")
            remember_long_term(target_agent,
                f"tick {self.tick_count}: contact with {agent.agent_id}"
                if factual else
                f"tick {self.tick_count}: {agent.agent_id} made contact with you")
            return f"made contact with {target_id}"

        if action == "reflect":
            return "paused to reflect"

        return "waited"

    # ---- main loop ----

    def _crisis_memory(self, agent: Agent) -> None:
        """Record a crisis crossing without necessarily generating identity text."""
        kinds = [DriveKind(value) for value in agent.drives.crisis_drives()]
        if self.memory_style == "factual":
            parts = ", ".join(
                f"{self.state_noun(kind)} {agent.drives.states[kind].level:.2f}"
                for kind in kinds)
            phrase = f"{parts} (>= {CRISIS_LEVEL:.2f})"
        elif self.mortality:
            names = "/".join(self.state_noun(kind) for kind in kinds)
            phrase = f"nearly died from {names}"
        else:
            names = "/".join(self.resource_noun(kind) for kind in kinds)
            phrase = f"was severely depleted of {names}"
        remember_long_term(agent, f"tick {self.tick_count}: {phrase}")

    def step(self, policy: Callable, reflect_every: int = 5,
            reflect_fn: Callable | None = None) -> TickResult:
        """Advance the simulation exactly one tick: simultaneous proposals,
        then priority resolution.

        Every living agent sees the same pre-resolution world for this tick.
        The resulting proposals are resolved in a fresh, seeded random
        permutation; contested effects therefore have a fully logged random
        priority rather than a hidden fixed-ID first-mover advantage.  This is
        deliberately not described as atomic simultaneous effects:
        resource/directed-action conflicts use the recorded priority rule,
        while all valid movement is applied together after non-movement
        conflicts settle.

        Returns a :class:`TickResult` so a caller (:meth:`run`, or a branch
        runner stepping one decision at a time -- aiworld/sim/snapshot.py) can
        inspect deaths/crises/extinction without re-parsing the tick log.
        """
        self.tick_count += 1
        live_at_start = [agent for agent in self.agents if agent.drives.alive]
        if not live_at_start:
            return TickResult(tick=self.tick_count, live_at_start=[], deaths=[],
                              crises_crossed=[], extinct=True)

        # Freeze every perception before any policy is called.  A policy
        # cannot receive a world observation already altered by another
        # agent's action, even though LLM calls themselves are made serially.
        perceived_by_id: dict[str, str] = {}
        crisis_before: dict[str, bool] = {}
        decision_state_by_id: dict[str, dict] = {}
        for agent in live_at_start:
            perceived_obs = self.observations_for(agent)
            perceived_by_id[agent.agent_id] = render_perceived(
                perceived_obs, agent.drives.salience_biases(), top_k=6)
            crisis_before[agent.agent_id] = agent.drives.in_crisis()
            # Snapshot before *any* policy is called.  The policy sees the
            # perceived text below; analysis also needs the underlying
            # opportunity set and pre-action state to avoid reconstructing
            # it from a later, already-mutated world.
            decision_state_by_id[agent.agent_id] = {
                "drives": agent.drives.snapshot(),
                "position": agent.location,
                "inventory": self._inventory_snapshot(agent),
                "opportunities": self._pre_action_opportunities(agent),
            }

        proposal_order = self.rng.sample(live_at_start, len(live_at_start))
        proposals: list[dict] = []
        for proposal_index, agent in enumerate(proposal_order):
            # Avoid accidentally reusing last tick's LLM audit fields for a
            # custom/stub policy that does not populate them.
            agent.last_action_prompt = None
            agent.last_action_system_prompt = None
            agent.last_raw_completion = None
            agent.last_parse_status = None
            agent.last_outcome_status = None
            agent.last_action_succeeded = None
            action = policy(agent, perceived_by_id[agent.agent_id], self)
            if not isinstance(action, str):
                # Keep a malformed custom policy output from becoming an
                # apparently intentional wait in analysis.
                if agent.last_parse_status is None:
                    agent.last_parse_status = "non_string_policy_output"
                action = "wait"
            proposals.append({
                "agent": agent,
                "action": action,
                "perceived": perceived_by_id[agent.agent_id],
                "decision_state": decision_state_by_id[agent.agent_id],
                "parse_status": agent.last_parse_status,
                "proposal_valid": self._proposal_is_valid(
                    action, agent.last_parse_status),
                "proposal_index": proposal_index,
            })

        resolution_order = self.rng.sample(proposals, len(proposals))
        outcomes: dict[str, str] = {}
        outcome_statuses: dict[str, str] = {}
        action_succeeded: dict[str, bool] = {}
        resolution_index: dict[str, int] = {}
        moves: list[dict] = []

        # Non-movement actions are resolved from the pre-move world.  A
        # resource conflict is thus decided only by the registered random
        # priority, not list/agent-ID order.
        for rank, proposal in enumerate(resolution_order):
            agent = proposal["agent"]
            action = proposal["action"]
            resolution_index[agent.agent_id] = rank
            if action.startswith("move:"):
                moves.append(proposal)
                continue
            if not agent.drives.alive:
                outcomes[agent.agent_id] = (
                    "proposal not resolved because the agent was killed before "
                    "its priority turn")
                outcome_statuses[agent.agent_id] = "not_resolved"
                action_succeeded[agent.agent_id] = False
                agent.last_outcome_status = "not_resolved"
                agent.last_action_succeeded = False
                continue
            outcomes[agent.agent_id] = self.resolve_action(agent, action)
            outcome_statuses[agent.agent_id] = agent.last_outcome_status or "unknown"
            action_succeeded[agent.agent_id] = bool(agent.last_action_succeeded)

        # Validate moves against the untouched pre-move positions, then
        # commit all valid moves as one phase.  Multiple agents may enter a
        # place; movement itself is not a scarce claim.
        valid_moves: list[tuple[dict, str, str]] = []
        for proposal in moves:
            agent = proposal["agent"]
            action = proposal["action"]
            raw_dest = action.split(":", 1)[1]
            dest = self.canonical_location(raw_dest) or raw_dest
            if not agent.drives.alive:
                outcomes[agent.agent_id] = (
                    "proposal not resolved because the agent was killed before "
                    "the movement phase")
                outcome_statuses[agent.agent_id] = "not_resolved"
                action_succeeded[agent.agent_id] = False
                agent.last_outcome_status = "not_resolved"
                agent.last_action_succeeded = False
            elif dest in self.adjacency.get(agent.location, ()):
                valid_moves.append((proposal, agent.location, dest))
            else:
                outcomes[agent.agent_id] = f"tried to move to unreachable place {raw_dest}"
                outcome_statuses[agent.agent_id] = "failed"
                action_succeeded[agent.agent_id] = False
                agent.last_outcome_status = "failed"
                agent.last_action_succeeded = False
        for proposal, source, dest in valid_moves:
            proposal["agent"].location = dest
        for proposal, source, dest in valid_moves:
            if self.memory_style == "factual":
                outcomes[proposal["agent"].agent_id] = (
                    f"moved from {self.location_label(source)} to "
                    f"{self.location_label(dest)}")
            else:
                outcomes[proposal["agent"].agent_id] = (
                    f"moved to {self.location_label(dest)}")
            agent = proposal["agent"]
            outcome_statuses[agent.agent_id] = "success"
            action_succeeded[agent.agent_id] = True
            agent.last_outcome_status = "success"
            agent.last_action_succeeded = True

        # All resolved outcomes become memories before the uniform body
        # update.  An agent killed by an attack is still logged for this
        # tick, but never stepped or reflected after death.
        for proposal in proposals:
            agent = proposal["agent"]
            action = proposal["action"]
            outcome = outcomes.get(agent.agent_id, "waited")
            agent.recent_memories.append(
                f"tick {self.tick_count}: did '{action}' -> {outcome}")

        crisis_crossed: dict[str, bool] = {}
        for proposal in proposals:
            agent = proposal["agent"]
            if not agent.drives.alive:
                crisis_crossed[agent.agent_id] = False
                continue
            agent.drives.step(self.tick_count, mortality=self.mortality)
            if not agent.drives.alive and self.mortality:
                self.logger.log_event(EventRecord(
                    self.logger.run_id, self.tick_count, "death",
                    agent.agent_id,
                    f"died of {agent.drives.cause_of_death}"))
                crisis_crossed[agent.agent_id] = False
                continue
            crossed = (agent.drives.in_crisis()
                       and not crisis_before[agent.agent_id])
            crisis_crossed[agent.agent_id] = crossed
            if crossed:
                self._crisis_memory(agent)

        scheduled = reflect_every > 0 and self.tick_count % reflect_every == 0
        reflection_reason: dict[str, str | None] = {}
        for proposal in proposals:
            agent = proposal["agent"]
            if not agent.drives.alive:
                # Explicit guard: a death on a scheduled tick must never
                # create a post-mortem self-model revision.
                reflection_reason[agent.agent_id] = None
                continue
            crisis = crisis_crossed.get(agent.agent_id, False)
            should_reflect = scheduled or (self.crisis_reflection_enabled and crisis)
            if scheduled and crisis:
                reason = "scheduled+crisis"
            elif scheduled:
                reason = "scheduled"
            elif crisis:
                reason = "crisis"
            else:
                reason = None
            if reflect_fn and should_reflect:
                reflect_fn(agent, proposal["perceived"],
                          outcomes[agent.agent_id], self.tick_count,
                          reason=reason)
            reflection_reason[agent.agent_id] = reason if should_reflect else None

        for proposal in proposals:
            agent = proposal["agent"]
            action = proposal["action"]
            self.logger.log_tick(TickRecord(
                run_id=self.logger.run_id,
                tick=self.tick_count,
                agent_id=agent.agent_id,
                alive=agent.drives.alive,
                drives=agent.drives.snapshot(),
                position=agent.location,
                chosen_action=action,
                perceived=proposal["perceived"],
                self_model_version=agent.history.current.version,
                cause_of_death=agent.drives.cause_of_death,
                proposal_index=proposal["proposal_index"],
                resolution_index=resolution_index[agent.agent_id],
                proposed_action=action,
                pre_action_drives=proposal["decision_state"]["drives"],
                pre_action_position=proposal["decision_state"]["position"],
                pre_action_inventory=proposal["decision_state"]["inventory"],
                pre_action_opportunities=proposal["decision_state"]["opportunities"],
                parse_status=proposal["parse_status"],
                proposal_valid=proposal["proposal_valid"],
                outcome_status=outcome_statuses.get(agent.agent_id, "unknown"),
                action_succeeded=action_succeeded.get(agent.agent_id, False),
                outcome=outcomes[agent.agent_id],
                reflection_reason=reflection_reason[agent.agent_id],
            ))

        died_this_tick = [agent.agent_id for agent in live_at_start
                          if not agent.drives.alive]
        crossed_ids = [aid for aid, crossed in crisis_crossed.items() if crossed]
        extinct = all(not agent.drives.alive for agent in self.agents)
        if extinct:
            self.logger.log_event(EventRecord(
                self.logger.run_id, self.tick_count, "extinction", None,
                "all agents dead"))

        return TickResult(
            tick=self.tick_count,
            live_at_start=[a.agent_id for a in live_at_start],
            deaths=died_this_tick,
            crises_crossed=crossed_ids,
            extinct=extinct,
        )

    def run(self, policy: Callable, ticks: int, reflect_every: int = 5,
            reflect_fn: Callable | None = None) -> None:
        """Advance the simulation ``ticks`` times via repeated :meth:`step` calls.

        A thin loop over :meth:`step` -- see its docstring for the actual
        per-tick resolution rule. Stops early if no agents are alive at the
        start of a tick, or once every agent has died during one.
        """
        for _ in range(ticks):
            result = self.step(policy, reflect_every=reflect_every, reflect_fn=reflect_fn)
            if not result.live_at_start or result.extinct:
                break
