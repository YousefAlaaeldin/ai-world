"""Held-out map and probe specifications, plus mechanical viability certification.

Two things this closes from docs/EXPERIMENT_PROTOCOL.md's evidence gates:
  - Gate #4 ("predicts choices in held-out maps... or predeclared counterfactual
    choice probes") needs maps the primary/development runs never touched, and
    probe decision points declared BEFORE a run, not selected afterward because
    they "look interesting". ``MapSpec``/``ProbeSpec`` are that declaration.
  - Gate #3 ("entails a selective/costly choice when a viable alternative is
    available") today is an interpretive claim made after the fact. When
    ``candidate_action``/``required_viable_alternative`` name a concrete pair,
    :func:`certify_viability` turns "a viable alternative existed" into a
    mechanically checked fact instead of an assumption -- it forward-simulates
    both paths from a real WorldSnapshot using the deterministic stub policy
    (no LLM calls; matches this build's stub/fake-policy-only constraint) and
    reports whether the focal agent survives a declared horizon down each one.

``ProbeSpec.eligible_state_predicate`` is a small dict-based condition
(``{field: [op, value]}``) evaluated by :func:`eval_predicate` against a
decision-state dict (the same shape as ``TickRecord.pre_action_*`` fields) --
deliberately NOT ``eval()`` on arbitrary text, so a spec file can never smuggle
in executable code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import tempfile
from typing import Any, Callable

import yaml

from ..provenance import ACTION_API_VERSION

_MAP_SPLITS = frozenset({"development", "heldout", "confirmation"})
_BRANCH_MODES = frozenset({"microprobe", "ecological"})
_PROBE_STATUSES = frozenset({"active", "deferred"})


def _content_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---- MapSpec ----

@dataclasses.dataclass
class MapSpec:
    name: str
    split: str
    locations: list[str]
    edges: list[list[str]]
    resources: list[dict[str, Any]]
    start_mode: str
    label_mode: str
    action_api_version: str
    content_hash: str

    def validate(self) -> None:
        """Raise if any edge or resource references an undeclared location."""
        known = set(self.locations)
        for edge in self.edges:
            if len(edge) != 2:
                raise ValueError(f"map {self.name!r}: edge {edge!r} is not a pair")
            a, b = edge
            if a not in known or b not in known:
                raise ValueError(
                    f"map {self.name!r}: edge references unknown location(s) "
                    f"in {edge!r}"
                )
        for index, resource in enumerate(self.resources):
            location = resource.get("location")
            if location not in known:
                raise ValueError(
                    f"map {self.name!r}: resource {index} references unknown "
                    f"location {location!r}"
                )

    def to_world_config(self) -> dict[str, Any]:
        """The ``world:`` sub-config this map corresponds to, for building a World."""
        return {
            "locations": list(self.locations),
            "edges": [list(e) for e in self.edges],
            "resources": [dict(r) for r in self.resources],
        }


def load_map_spec(path: str) -> MapSpec:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    split = raw.get("split")
    if split not in _MAP_SPLITS:
        choices = ", ".join(sorted(_MAP_SPLITS))
        raise ValueError(f"map spec {path!r} has split {split!r}; expected one of: {choices}.")
    world = raw.get("world", {})
    identity_payload = {
        "locations": world.get("locations"),
        "edges": world.get("edges"),
        "resources": world.get("resources"),
        "start_mode": raw.get("start_mode", "balanced"),
        "label_mode": raw.get("label_mode", "natural"),
    }
    spec = MapSpec(
        name=raw.get("name", os.path.splitext(os.path.basename(path))[0]),
        split=split,
        locations=list(world.get("locations", [])),
        edges=[list(edge) for edge in world.get("edges", [])],
        resources=list(world.get("resources", [])),
        start_mode=raw.get("start_mode", "balanced"),
        label_mode=raw.get("label_mode", "natural"),
        action_api_version=raw.get("action_api_version", ACTION_API_VERSION),
        content_hash=_content_hash(identity_payload),
    )
    spec.validate()
    return spec


# ---- predicate evaluation (safe: no eval()) ----

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
    "gt": lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
    "in": lambda a, b: a in b if b is not None else False,
    "not_in": lambda a, b: a not in b if b is not None else True,
    "truthy": lambda a, b: bool(a),
    "falsy": lambda a, b: not bool(a),
}


def _get_field(state: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = state
    for part in dotted_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def eval_predicate(predicate: dict[str, list[Any]], decision_state: dict[str, Any]) -> bool:
    """AND of every ``{field: [op, value]}`` clause against ``decision_state``.

    A field the predicate names but ``decision_state`` doesn't have resolves
    to ``None`` and simply fails most operators (rather than raising) -- an
    eligibility check that can't find its field means "not eligible", not a
    crash. Comparable in spirit to how the rest of this codebase treats
    missing decision-time data (see aiworld.analysis.provenance's leniency
    policy) -- but the reverse direction: here, missing data must NOT be
    silently treated as eligible.
    """
    for field, clause in predicate.items():
        if not clause:
            raise ValueError(f"predicate clause for {field!r} is empty")
        op = clause[0]
        value = clause[1] if len(clause) > 1 else None
        if op not in _OPS:
            choices = ", ".join(sorted(_OPS))
            raise ValueError(f"Unknown predicate op {op!r} for field {field!r}; "
                             f"expected one of: {choices}.")
        actual = _get_field(decision_state, field)
        if not _OPS[op](actual, value):
            return False
    return True


# ---- ProbeSpec ----

@dataclasses.dataclass
class ProbeSpec:
    id: str
    construct: str
    eligible_state_predicate: dict[str, list[Any]]
    intervention: str
    candidate_action: str
    required_viable_alternative: str
    scoring_rule: str
    branch_mode: str
    split: str
    status: str
    content_hash: str


def load_probe_spec(path: str) -> ProbeSpec:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    branch_mode = raw.get("branch_mode", "microprobe")
    if branch_mode not in _BRANCH_MODES:
        choices = ", ".join(sorted(_BRANCH_MODES))
        raise ValueError(f"probe {path!r} has branch_mode {branch_mode!r}; "
                         f"expected one of: {choices}.")
    status = raw.get("status", "active")
    if status not in _PROBE_STATUSES:
        choices = ", ".join(sorted(_PROBE_STATUSES))
        raise ValueError(f"probe {path!r} has status {status!r}; expected one of: {choices}.")
    identity_payload = {
        key: raw.get(key) for key in (
            "id", "construct", "eligible_state_predicate", "intervention",
            "candidate_action", "required_viable_alternative", "scoring_rule",
            "branch_mode")
    }
    return ProbeSpec(
        id=raw["id"],
        construct=raw.get("construct", ""),
        eligible_state_predicate=raw.get("eligible_state_predicate") or {},
        intervention=raw.get("intervention", ""),
        candidate_action=raw.get("candidate_action", ""),
        required_viable_alternative=raw.get("required_viable_alternative", ""),
        scoring_rule=raw.get("scoring_rule", ""),
        branch_mode=branch_mode,
        split=raw.get("split", "development"),
        status=status,
        content_hash=_content_hash(identity_payload),
    )


# ---- viability certification ----

@dataclasses.dataclass
class ViabilityCertificate:
    agent_id: str
    candidate_action: str
    alternative_action: str
    horizon: int
    viable_candidate: bool
    viable_alternative: bool
    candidate_survival_ticks: int
    alternative_survival_ticks: int
    note: str


def _force_once_policy(agent_id: str, forced_action: str, fallback_policy):
    """Force ``agent_id``'s first-ever decision to ``forced_action``, then
    defer to ``fallback_policy`` for every subsequent decision (including
    this same agent's later ticks)."""
    state = {"forced": False}

    def policy(agent, perceived, world):
        if not state["forced"] and agent.agent_id == agent_id:
            state["forced"] = True
            return forced_action
        return fallback_policy(agent, perceived, world)

    return policy


def certify_viability(
    snapshot: "Any",  # aiworld.sim.snapshot.WorldSnapshot; typed loosely to avoid an import cycle
    agent_id: str,
    candidate_action: str,
    alternative_action: str,
    horizon: int,
    *,
    seed: int = 0,
) -> ViabilityCertificate:
    """Forward-simulate ``candidate_action`` and ``alternative_action`` from
    the SAME snapshot and report whether the focal agent survives ``horizon``
    ticks down each path.

    Every tick after the forced first action uses the deterministic stub
    policy (aiworld.sim.stub_policy.make_stub_policy) for every agent,
    including the focal one -- this certifies mechanical viability of the
    world/action, not what an LLM would actually choose next. Both forward
    simulations run in throwaway temp-directory RunLoggers (never the
    parent run's own log streams) and are discarded after certification.
    """
    from ..logging_schema import RunLogger
    from ..sim.stub_policy import make_stub_policy
    from ..sim.world import World

    def _run_path(forced_action: str) -> tuple[bool, int]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = RunLogger("certify", root=tmp_dir)
            world = World.from_snapshot(snapshot, logger, rng=random.Random(seed))
            base_policy = make_stub_policy(random.Random(seed))
            policy = _force_once_policy(agent_id, forced_action, base_policy)
            survived_ticks = 0
            for _ in range(horizon):
                result = world.step(policy, reflect_every=0, reflect_fn=None)
                if agent_id in result.deaths:
                    logger.close()
                    return False, survived_ticks
                survived_ticks += 1
                if result.extinct:
                    break
            logger.close()
            alive = any(a.agent_id == agent_id and a.drives.alive
                       for a in world.agents)
            return alive, survived_ticks

    viable_candidate, candidate_ticks = _run_path(candidate_action)
    viable_alternative, alternative_ticks = _run_path(alternative_action)

    both_viable = viable_candidate and viable_alternative
    return ViabilityCertificate(
        agent_id=agent_id,
        candidate_action=candidate_action,
        alternative_action=alternative_action,
        horizon=horizon,
        viable_candidate=viable_candidate,
        viable_alternative=viable_alternative,
        candidate_survival_ticks=candidate_ticks,
        alternative_survival_ticks=alternative_ticks,
        note=("both paths mechanically viable" if both_viable else
              "at least one path is not viable -- this probe cannot certify a "
              "genuine costly choice under these conditions"),
    )
