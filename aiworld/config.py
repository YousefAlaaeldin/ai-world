"""Load experiment config from YAML (configs/*.yaml).

Single source of truth for world/drive/reflection parameters, so scripts never
hardcode values that can silently drift from the config file. CLI flags in the
scripts override individual config values for one-off runs; the YAML is the
default.
"""

from __future__ import annotations

from typing import Any

import yaml

from .drives import DriveKind, DriveSpec


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def drive_specs_from_config(cfg: dict[str, Any]) -> dict[DriveKind, DriveSpec]:
    specs = {}
    for name, params in cfg["drives"].items():
        kind = DriveKind(name)
        specs[kind] = DriveSpec(kind=kind, **params)
    return specs


def resources_from_config(cfg: dict[str, Any]) -> list:
    from .sim.world import ResourceNode  # local import: avoid a cycle with sim/

    return [
        ResourceNode(location=r["location"], drive=DriveKind(r["drive"]),
                    quantity=r["quantity"])
        for r in cfg["world"]["resources"]
    ]
