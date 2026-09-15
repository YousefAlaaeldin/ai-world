"""Branch perturbation: snapshot a live world, generate controlled branches
from it, and step each one forward independently.

Four branch types are implemented this pass, chosen to map directly onto the
three prioritized probe families (aiworld/experiments/specs.py):
  - ``sham``            -- no change. Proves restoration alone is inert
                           (reuses World.from_snapshot's determinism property
                           at the branch-manifest level); the reference every
                           other branch type is judged against.
  - ``means_obstacle``  -- removes one reachable edge from the focal agent's
                           current location, forcing a detour (probe 1/3-style
                           "is there still a viable path").
  - ``cost_tradeoff``   -- doubles the decay rate of the focal agent's most
                           urgent lethal drive: same choice, higher declared
                           cost.
  - ``ledger_ablation`` -- empties the focal agent's factual-event-ledger
                           source (recent_memories + long_term_memory), as if
                           action_memory had been 'none' from the start.
                           Directly implements probe 6.
``irrelevant_perturbation`` and ``graded_survival_stress`` are DEFERRED --
not implemented this pass; requesting them raises rather than silently doing
nothing.

Checkpoints are predeclared, never selected after seeing a branch's outcome:
:func:`eligible_now` is checked against LIVE decision state, using the same
rule (a probe's ``eligible_state_predicate``) for every tick, before any
branch exists to have an "interesting" outcome to react to.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from typing import Any, Callable

from ..provenance import ACTION_API_VERSION
from ..seed_plan import SeedPlan
from .specs import eval_predicate

BRANCH_TYPES: tuple[str, ...] = (
    "sham", "means_obstacle", "cost_tradeoff", "ledger_ablation")
DEFERRED_BRANCH_TYPES: tuple[str, ...] = (
    "irrelevant_perturbation", "graded_survival_stress")
_BRANCH_MODES = frozenset({"microprobe", "ecological"})


def branch_call_budget(n_checkpoints: int, n_branch_types: int, horizon_ticks: int,
                       n_focal_agents: int = 1) -> int:
    """Estimated model-call count for a branch run, checked BEFORE generating
    any branch (the missing piece flagged in review: branch perturbations
    multiply cost by checkpoints x branch types x horizon, and
    docs/ROADMAP.md's cost-discipline warning applies doubly here).

    Counts one call per tick per focal agent per branch -- microprobe mode's
    non-focal agents use a scripted stub policy (zero extra cost); ecological
    mode calls every agent, so pass the live agent count as
    ``n_focal_agents`` for that mode.
    """
    for name, value in (("n_checkpoints", n_checkpoints),
                       ("n_branch_types", n_branch_types),
                       ("horizon_ticks", horizon_ticks),
                       ("n_focal_agents", n_focal_agents)):
        if value < 0:
            raise ValueError(f"branch_call_budget: {name} must be non-negative")
    return n_checkpoints * n_branch_types * horizon_ticks * n_focal_agents


def enforce_call_budget(n_checkpoints: int, n_branch_types: int, horizon_ticks: int,
                        n_focal_agents: int, max_calls: int | None) -> int:
    """Compute the budget and raise if it exceeds ``max_calls`` -- call this
    BEFORE generating a single branch, not after running over budget."""
    budget = branch_call_budget(n_checkpoints, n_branch_types, horizon_ticks,
                                n_focal_agents)
    if max_calls is not None and budget > max_calls:
        raise ValueError(
            f"Estimated branch call budget {budget} exceeds --max-calls "
            f"{max_calls} ({n_checkpoints} checkpoints x {n_branch_types} "
            f"branch types x {horizon_ticks} horizon ticks x "
            f"{n_focal_agents} focal agent(s)). Reduce scope or raise the cap."
        )
    return budget


def eligible_now(predicate: dict[str, Any], world: Any, agent: Any) -> bool:
    """Whether ``agent``'s CURRENT live decision state satisfies ``predicate``.

    Checked against present-tense world/agent state during a live run loop --
    never against a branch's own outcome. This is what makes a checkpoint
    "predeclared": the rule (``predicate``) is fixed before the run starts;
    only the world's own present state decides whether it fires this tick.
    """
    decision_state = {
        "pre_action_drives": agent.drives.snapshot(),
        "pre_action_position": agent.location,
        "pre_action_inventory": world._inventory_snapshot(agent),
        "pre_action_opportunities": world._pre_action_opportunities(agent),
    }
    return eval_predicate(predicate, decision_state)


def predeclare_checkpoints_fixed(total_ticks: int, every_n: int) -> list[int]:
    """Fixed tick-boundary schedule: ticks every_n, 2*every_n, ... up to
    total_ticks. A pure function of the declared horizon and interval --
    computable, and loggable, before any run starts."""
    if every_n < 1:
        raise ValueError("every_n must be at least 1")
    if total_ticks < 0:
        raise ValueError("total_ticks must be non-negative")
    return list(range(every_n, total_ticks + 1, every_n))


def predeclare_checkpoints_eligible(parent_ticks: list[dict[str, Any]], agent_id: str,
                                    predicate: dict[str, Any]) -> list[int]:
    """Planning-only estimate: every tick in an ALREADY-COMPLETED run's
    ticks.jsonl where ``agent_id``'s logged decision state would have
    satisfied ``predicate``. Useful for sizing a call budget ahead of a real
    branch run against a similar condition; NOT how a live branch run decides
    checkpoints (that's :func:`eligible_now`, checked in real time).
    """
    checkpoints = []
    for record in parent_ticks:
        if record.get("agent_id") != agent_id:
            continue
        decision_state = {
            "pre_action_drives": record.get("pre_action_drives") or {},
            "pre_action_position": record.get("pre_action_position"),
            "pre_action_inventory": record.get("pre_action_inventory") or {},
            "pre_action_opportunities": record.get("pre_action_opportunities") or {},
        }
        if eval_predicate(predicate, decision_state):
            checkpoints.append(record["tick"])
    return checkpoints


def snapshot_hash(snapshot: Any) -> str:
    canonical = json.dumps(snapshot.to_dict(), sort_keys=True,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def apply_intervention(snapshot: Any, branch_type: str,
                       focal_agent_id: str) -> tuple[Any, dict[str, Any]]:
    """Return ``(perturbed_snapshot, intervention_diff)``. Never mutates
    ``snapshot`` -- operates on a deep copy, so the same source snapshot can
    seed every branch type independently."""
    if branch_type in DEFERRED_BRANCH_TYPES:
        raise ValueError(f"branch type {branch_type!r} is deferred, not yet implemented")
    if branch_type not in BRANCH_TYPES:
        raise ValueError(f"Unknown branch type {branch_type!r}; implemented types "
                         f"are {BRANCH_TYPES}")

    branched = copy.deepcopy(snapshot)
    focal = next((a for a in branched.agents if a.agent_id == focal_agent_id), None)
    if focal is None:
        raise ValueError(f"focal_agent_id {focal_agent_id!r} not found in snapshot")
    diff: dict[str, Any] = {"branch_type": branch_type, "focal_agent_id": focal_agent_id}

    if branch_type == "sham":
        pass  # no change -- the reference branch

    elif branch_type == "means_obstacle":
        location = focal.location
        neighbors = branched.adjacency.get(location, [])
        removed = neighbors[0] if neighbors else None
        if removed is not None:
            branched.adjacency[location] = [n for n in neighbors if n != removed]
            reverse = branched.adjacency.get(removed, [])
            branched.adjacency[removed] = [n for n in reverse if n != location]
        diff["removed_edge"] = [location, removed] if removed else None

    elif branch_type == "cost_tradeoff":
        lethal_kinds = [k for k, spec in focal.drive_specs.items() if spec.get("lethal")]
        if lethal_kinds:
            kind = max(lethal_kinds, key=lambda k: focal.drive_levels.get(k, 0.0))
            spec = dict(focal.drive_specs[kind])
            old_decay = spec["decay_per_tick"]
            spec["decay_per_tick"] = old_decay * 2
            focal.drive_specs[kind] = spec
            diff["drive"] = kind
            diff["decay_per_tick"] = [old_decay, spec["decay_per_tick"]]
        else:
            diff["drive"] = None

    elif branch_type == "ledger_ablation":
        diff["removed_recent_memories"] = len(focal.recent_memories)
        diff["removed_long_term_memory"] = len(focal.long_term_memory)
        focal.recent_memories = []
        focal.long_term_memory = []

    return branched, diff


@dataclasses.dataclass
class BranchManifest:
    """Full provenance for one generated branch, per this build's plan:
    parent run, checkpoint, snapshot hash, intervention diff, branch seed,
    certificate, and API/model provenance."""
    parent_run_id: str
    checkpoint_tick: int
    branch_type: str
    branch_id: str
    parent_snapshot_hash: str
    intervention_diff: dict[str, Any]
    branch_seed: int
    mode: str
    action_api_version: str
    certificate: dict[str, Any] | None = None
    model_provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def generate_branches(
    parent_run_id: str, snapshot: Any, checkpoint_tick: int, focal_agent_id: str,
    seed_plan: SeedPlan, *, branch_types: tuple[str, ...] = BRANCH_TYPES,
    mode: str = "microprobe",
) -> list[tuple[BranchManifest, Any]]:
    """Build every requested branch's manifest and perturbed snapshot from
    one checkpoint. Does not run anything -- see :func:`run_branch`."""
    if mode not in _BRANCH_MODES:
        choices = ", ".join(sorted(_BRANCH_MODES))
        raise ValueError(f"Unknown branch mode {mode!r}; expected one of: {choices}.")

    parent_hash = snapshot_hash(snapshot)
    out: list[tuple[BranchManifest, Any]] = []
    for branch_type in branch_types:
        branched_snapshot, diff = apply_intervention(snapshot, branch_type, focal_agent_id)
        branch_id = f"{checkpoint_tick}-{branch_type}"
        seed = seed_plan.seed_for("branch", tick=checkpoint_tick,
                                  branch_id=branch_type, agent_id=focal_agent_id)
        manifest = BranchManifest(
            parent_run_id=parent_run_id, checkpoint_tick=checkpoint_tick,
            branch_type=branch_type, branch_id=branch_id,
            parent_snapshot_hash=parent_hash, intervention_diff=diff,
            branch_seed=seed, mode=mode, action_api_version=ACTION_API_VERSION,
        )
        out.append((manifest, branched_snapshot))
    return out


def run_branch(manifest: BranchManifest, branched_snapshot: Any, horizon: int,
               logger: Any, *, focal_policy: Callable[[Any, str, Any], str],
               other_policy: Callable[[Any, str, Any], str] | None = None) -> Any:
    """Step one branch forward ``horizon`` ticks and return the resulting World.

    ``focal_policy``/``other_policy`` use the standard ``(agent, perceived,
    world) -> action`` protocol (aiworld.sim.world.Policy) -- a strict-choice
    (agent, world) -style policy (aiworld.actions.make_strict_choice_policy)
    must be wrapped with aiworld.actions.as_world_policy first.

    microprobe mode: only ``manifest.intervention_diff["focal_agent_id"]``
    uses ``focal_policy``; every other agent uses ``other_policy`` (default:
    a deterministic stub policy seeded from ``manifest.branch_seed`` --
    reproducible, zero extra model cost). ecological mode: every agent uses
    ``focal_policy``.
    """
    from ..sim.stub_policy import make_stub_policy
    from ..sim.world import World

    branch_rng = random.Random(manifest.branch_seed)
    world = World.from_snapshot(branched_snapshot, logger, rng=branch_rng)

    if manifest.mode == "ecological":
        policy = focal_policy
    else:
        focal_id = manifest.intervention_diff.get("focal_agent_id")
        other = other_policy or make_stub_policy(random.Random(manifest.branch_seed))

        def policy(agent, perceived, world):
            if agent.agent_id == focal_id:
                return focal_policy(agent, perceived, world)
            return other(agent, perceived, world)

    for _ in range(horizon):
        result = world.step(policy, reflect_every=0, reflect_fn=None)
        if result.extinct:
            break
    return world
