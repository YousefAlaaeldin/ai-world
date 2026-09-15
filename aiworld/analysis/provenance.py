"""Guards that keep incomplete or incompatible runs out of analysis.

Two failure modes this closes:
  1. A run that crashed, was aborted, or ended on invalid model output has a
     meta.json (written *before* the run body executes -- see
     logging_schema.py) but a truncated ticks.jsonl. Nothing before this
     module distinguished that from a short-but-complete run.
  2. Two sweeps compared as a paired contrast
     (aiworld.analysis.run_level.paired_condition_comparison) used a
     different model, decode setting, map mode, or start-allocation mode
     without anyone noticing, because nothing checked.

Leniency policy: a MISSING status.json or provenance field is treated as
unknown-but-permitted, never rejected. This matches the rest of the codebase
(e.g. aiworld.analysis.behavior_tags._is_parse_failure: "missing status means
either an old record or a custom policy, neither of which should be
retroactively labeled a failure") and keeps every hand-built test/analysis
fixture that predates RunLogger.finalize() working unchanged. Real runs
produced by the current runner scripts always get a status.json (finalize()
runs from a try/finally around the run body -- see scripts/run_llm.py), so
this still catches the common failure path (an exception during the run, or
an explicit invalid_output abort). It does NOT catch a hard-killed process
that never reaches its finally block; that would need per-tick journaling,
out of scope here.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..provenance import ACTION_API_VERSION, ANALYSIS_SCHEMA_VERSION


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_run_provenance(run_dir: str) -> dict[str, Any]:
    """meta.json (pre-run) merged with status.json (post-run), when present.

    ``status.json`` fields win on key collision -- it is the later, more
    authoritative record for anything both files happen to carry.
    """
    meta = _load_json(os.path.join(run_dir, "meta.json")) or {}
    status = _load_json(os.path.join(run_dir, "status.json")) or {}
    return {**meta, **status}


def require_completed_run(
    run_dir: str, *,
    expected_action_api_version: str = ACTION_API_VERSION,
    expected_analysis_schema_version: int = ANALYSIS_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Raise if ``run_dir`` explicitly declares itself incomplete or incompatible.

    See the module docstring for the leniency policy on missing data.
    """
    provenance = load_run_provenance(run_dir)
    status = provenance.get("status")
    if status is not None and status != "completed":
        raise ValueError(
            f"{run_dir!r} has status {status!r}, not 'completed'; it cannot "
            "enter analysis."
        )
    action_api_version = provenance.get("action_api_version")
    if (action_api_version is not None
            and action_api_version != expected_action_api_version):
        raise ValueError(
            f"{run_dir!r} was logged under action_api_version "
            f"{action_api_version!r}, incompatible with the analysis code's "
            f"{expected_action_api_version!r}."
        )
    analysis_schema_version = provenance.get("analysis_schema_version")
    if (analysis_schema_version is not None
            and analysis_schema_version != expected_analysis_schema_version):
        raise ValueError(
            f"{run_dir!r} was logged under analysis_schema_version "
            f"{analysis_schema_version!r}, incompatible with the analysis "
            f"code's {expected_analysis_schema_version!r}."
        )
    return provenance


# (path-into-provenance-dict) pairs checked for a paired condition comparison.
# Deliberately excludes fields that are SUPPOSED to differ per seed by design
# (exact resource_map / initial_locations placements) -- only the declared
# MODE has to match between two sweeps being compared.
_PAIR_COMPARABILITY_KEYS: tuple[tuple[str, ...], ...] = (
    ("llm", "model"),
    ("llm", "provider"),
    ("llm", "temperature"),
    ("resource_map_mode",),
    ("initial_location_mode",),
    ("action_api_version",),
    ("analysis_schema_version",),
)


def _get_path(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def require_compatible_pair(left_dir: str, right_dir: str, *,
                            require_completed: bool = True) -> None:
    """Raise unless two run directories share model/decode/map/start provenance.

    Guards aiworld.analysis.run_level.paired_condition_comparison against
    silently comparing sweeps that differ in more than the intended
    experimental factor. A field missing on either side is not evidence of
    incompatibility (see module docstring) and is skipped, not raised on.
    """
    left = (require_completed_run(left_dir) if require_completed
            else load_run_provenance(left_dir))
    right = (require_completed_run(right_dir) if require_completed
             else load_run_provenance(right_dir))
    mismatches = []
    for path in _PAIR_COMPARABILITY_KEYS:
        left_value = _get_path(left, path)
        right_value = _get_path(right, path)
        if left_value is None or right_value is None:
            continue
        if left_value != right_value:
            mismatches.append((".".join(path), left_value, right_value))
    if mismatches:
        detail = "; ".join(f"{key}: {lv!r} != {rv!r}" for key, lv, rv in mismatches)
        raise ValueError(
            f"{left_dir!r} and {right_dir!r} are not a compatible pair: {detail}"
        )
