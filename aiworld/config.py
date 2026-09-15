"""Load experiment config from YAML (configs/*.yaml).

Single source of truth for world/drive/reflection parameters, so scripts never
hardcode values that can silently drift from the config file. CLI flags in the
scripts override individual config values for one-off runs; the YAML is the
default.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

from .drives import DriveKind, DriveSpec


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_PAIRED_CONFIG_IGNORED_TOP_LEVEL_KEYS = frozenset({"experiment_name", "description"})


def _flatten_config(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten_config(value, path))
        else:
            flat[path] = value
    return flat


def validate_paired_configs(config_a: Mapping[str, Any], config_b: Mapping[str, Any],
                            declared_factor: str) -> None:
    """Raise unless two resolved condition configs differ ONLY in
    ``declared_factor`` (a dotted path, e.g. ``"world.mortality"``; a path
    naming a sub-tree, e.g. ``"world"``, allows every key beneath it to
    differ).

    Enforces the "paired configs must differ only in their declared factor"
    requirement from this build's Phase 7 condition matrix -- a single-factor
    contrast that quietly also differs in, say, resource quantities or
    reflection cadence is not actually a single-factor contrast.
    ``experiment_name``/``description`` are ignored: they are expected to
    differ between any two named conditions and carry no mechanics.
    """
    flat_a = _flatten_config(
        {k: v for k, v in config_a.items() if k not in _PAIRED_CONFIG_IGNORED_TOP_LEVEL_KEYS})
    flat_b = _flatten_config(
        {k: v for k, v in config_b.items() if k not in _PAIRED_CONFIG_IGNORED_TOP_LEVEL_KEYS})

    def _under_factor(key: str) -> bool:
        return key == declared_factor or key.startswith(declared_factor + ".")

    found = (any(_under_factor(k) for k in flat_a)
            or any(_under_factor(k) for k in flat_b))
    if not found:
        raise ValueError(
            f"declared_factor {declared_factor!r} was not found in either config."
        )

    mismatches = []
    for key in sorted(set(flat_a) | set(flat_b)):
        if _under_factor(key):
            continue
        value_a = flat_a.get(key, "<missing>")
        value_b = flat_b.get(key, "<missing>")
        if value_a != value_b:
            mismatches.append((key, value_a, value_b))

    if mismatches:
        detail = "; ".join(f"{key}: {a!r} != {b!r}" for key, a, b in mismatches)
        raise ValueError(
            f"Paired configs differ in more than the declared factor "
            f"{declared_factor!r}: {detail}"
        )


def drive_specs_from_config(cfg: dict[str, Any]) -> dict[DriveKind, DriveSpec]:
    specs = {}
    for name, params in cfg["drives"].items():
        kind = DriveKind(name)
        specs[kind] = DriveSpec(kind=kind, **params)
    return specs


_START_LOCATION_MODES = frozenset({"legacy", "balanced"})
_RESOURCE_MAP_MODES = frozenset({"fixed", "counterbalanced"})


def _validate_mode(mode: str, allowed: frozenset[str], label: str) -> None:
    """Raise a useful error instead of silently accepting a misspelled arm."""
    if mode not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown {label} {mode!r}; expected one of: {choices}.")


def _resource_nodes_from_config(
    cfg: dict[str, Any], location_mapping: Mapping[str, str] | None = None,
) -> list:
    """Instantiate configured resources, optionally after a location relabeling."""
    from .sim.world import ResourceNode  # local import: avoid a cycle with sim/

    location_mapping = location_mapping or {}
    return [
        ResourceNode(location=location_mapping.get(r["location"], r["location"]),
                     drive=DriveKind(r["drive"]), quantity=r["quantity"])
        for r in cfg["world"]["resources"]
    ]


def resource_location_mapping(
    cfg: dict[str, Any],
    rng: random.Random | None = None,
    map_mode: str = "fixed",
) -> dict[str, str]:
    """Return the configured-location -> effective-location map for one run.

    ``fixed`` is the legacy condition: every configured location maps to itself.
    ``counterbalanced`` draws a deterministic (given ``rng``) permutation within
    each *degree class* of the configured graph.  A resource can therefore move
    between interchangeable leaves (for example, the four leaves of a hub) but
    never from a leaf to a hub.  The world topology itself is not changed.

    A caller that needs both the instantiated resources and the mapping metadata
    should prefer :func:`resource_map_from_config`, which makes the draw exactly
    once and returns both together.
    """
    _validate_mode(map_mode, _RESOURCE_MAP_MODES, "resource map mode")

    world = cfg.get("world", {})
    if map_mode == "fixed":
        # Keep this permissive so the legacy resources_from_config(cfg) call has
        # the same requirements it had before this counterbalancing feature.
        locations = list(world.get("locations", ()))
        for resource in world.get("resources", ()):
            location = resource.get("location")
            if location is not None and location not in locations:
                locations.append(location)
        return {location: location for location in locations}

    if rng is None:
        raise ValueError(
            "resource map mode 'counterbalanced' requires a supplied random.Random "
            "instance so the placement is reproducible."
        )

    try:
        locations = list(world["locations"])
    except KeyError as exc:
        raise ValueError(
            "resource map mode 'counterbalanced' requires world.locations."
        ) from exc
    if not locations:
        raise ValueError(
            "resource map mode 'counterbalanced' requires at least one world location."
        )
    if len(set(locations)) != len(locations):
        raise ValueError(
            "resource map mode 'counterbalanced' requires unique world.locations."
        )

    # Validate resource references before asking adjacency_from_config() to form
    # degree classes.  This turns a later, cryptic KeyError into a config error
    # that identifies the offending resource.
    known_locations = set(locations)
    for index, resource in enumerate(world.get("resources", ())):
        location = resource.get("location")
        if location not in known_locations:
            raise ValueError(
                "resource map mode 'counterbalanced' found resource "
                f"{index} at unknown location {location!r}."
            )

    adjacency = adjacency_from_config(cfg)
    degree_classes: dict[int, list[str]] = {}
    for location in locations:
        degree_classes.setdefault(len(adjacency[location]), []).append(location)

    mapping: dict[str, str] = {}
    # Sorting degree keys makes the number/order of RNG draws independent of the
    # incidental insertion order of the dictionary.  Within a degree class,
    # location order remains the configured order, which is part of the map spec.
    for degree in sorted(degree_classes):
        sources = degree_classes[degree]
        destinations = list(sources)
        rng.shuffle(destinations)
        mapping.update(zip(sources, destinations, strict=True))
    return mapping


def resource_map_metadata(
    resources: Sequence[Any],
    *,
    map_mode: str = "fixed",
    location_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe effective resource placements in JSON-safe, stable form.

    The returned ``map_id`` is a short content hash of the declared map mode,
    mapping, and effective placements.  Put this object in run metadata so a
    later analysis can group or rule out map-specific effects without trying to
    reconstruct the PRNG state.
    """
    _validate_mode(map_mode, _RESOURCE_MAP_MODES, "resource map mode")
    mapping = dict(location_mapping or {})
    placements = [
        {
            "location": resource.location,
            "drive": (resource.drive.value
                      if isinstance(resource.drive, DriveKind)
                      else str(resource.drive)),
            "quantity": resource.quantity,
        }
        for resource in resources
    ]
    identity_payload = {
        "map_mode": map_mode,
        "location_mapping": mapping,
        "resource_placements": placements,
    }
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    map_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {
        **identity_payload,
        "map_id": map_id,
    }


def resource_map_from_config(
    cfg: dict[str, Any],
    rng: random.Random | None = None,
    map_mode: str = "fixed",
) -> tuple[list, dict[str, Any]]:
    """Build resources plus reproducibility metadata for a condition/run.

    This is the runner-facing helper for T5 counterbalancing.  It performs one
    deterministic randomization draw, then returns the resulting ``ResourceNode``
    list and a metadata object containing the source-to-effective location map,
    effective placements, and a stable ``map_id``.
    """
    mapping = resource_location_mapping(cfg, rng=rng, map_mode=map_mode)
    resources = _resource_nodes_from_config(cfg, mapping)
    return resources, resource_map_metadata(
        resources, map_mode=map_mode, location_mapping=mapping)


def resources_from_config(
    cfg: dict[str, Any],
    rng: random.Random | None = None,
    map_mode: str = "fixed",
) -> list:
    """Instantiate configured resources.

    With the default ``map_mode='fixed'``, this is intentionally identical to
    the original API and does not consume ``rng``.  ``counterbalanced`` remaps
    resource locations only inside topology-equivalent degree classes; supply a
    seeded ``random.Random`` so that the placement can be reproduced.
    """
    _validate_mode(map_mode, _RESOURCE_MAP_MODES, "resource map mode")
    if map_mode == "fixed":
        return _resource_nodes_from_config(cfg)
    mapping = resource_location_mapping(cfg, rng=rng, map_mode=map_mode)
    return _resource_nodes_from_config(cfg, mapping)


def adjacency_from_config(cfg: dict[str, Any]) -> dict[str, set[str]]:
    """Build a bidirectional adjacency map from world.edges (pairs of location
    names). Falls back to fully connected (every location reachable from every
    other in one move) if `edges` is omitted -- kept for backward compatibility,
    but real configs should define edges: a fully-connected world flattens
    "environment" to a single binary (resource here or not) and gives movement
    no cost or structure to actually matter.
    """
    locations = cfg["world"]["locations"]
    edges = cfg["world"].get("edges")
    if not edges:
        return {loc: {other for other in locations if other != loc} for loc in locations}
    adjacency: dict[str, set[str]] = {loc: set() for loc in locations}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def initial_locations(
    rng: random.Random,
    locations: list[str],
    n_agents: int,
    mode: str = "legacy",
) -> list[str]:
    """Starting location per agent, in agent order.

    ``legacy`` preserves the original behavior exactly: when there are at least
    as many locations as agents, place them one-per-location; otherwise draw
    independently with replacement.  This matters for isolation conditions
    (configs/experiment_null_baseline.yaml), where NOT co-locating agents is
    the entire point.

    ``balanced`` is the T5 counterbalance: every location receives either
    ``floor(n_agents / n_locations)`` or ``ceil(...)`` agents, so placement
    counts differ by at most one.  The slots are shuffled before assignment, so
    no agent ID is structurally tied to a particular start location.
    """
    _validate_mode(mode, _START_LOCATION_MODES, "initial location mode")
    if mode == "legacy":
        if len(locations) >= n_agents:
            return rng.sample(locations, n_agents)
        return [rng.choice(locations) for _ in range(n_agents)]

    if n_agents < 0:
        raise ValueError("initial location mode 'balanced' requires n_agents >= 0.")
    if not locations:
        if n_agents == 0:
            return []
        raise ValueError(
            "initial location mode 'balanced' requires at least one location "
            "when n_agents is positive."
        )
    if len(set(locations)) != len(locations):
        raise ValueError(
            "initial location mode 'balanced' requires unique locations."
        )

    # Shuffle locations first to decide which ones receive the unavoidable
    # remainder, then shuffle the resulting slots to randomize agent-ID-to-place
    # assignment.  Both operations use the supplied RNG, making a run replayable.
    shuffled_locations = list(locations)
    rng.shuffle(shuffled_locations)
    minimum, remainder = divmod(n_agents, len(shuffled_locations))
    slots = [
        location
        for index, location in enumerate(shuffled_locations)
        for _ in range(minimum + (1 if index < remainder else 0))
    ]
    rng.shuffle(slots)
    return slots
