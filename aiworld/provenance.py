"""Version constants and lightweight provenance helpers shared by runners and
analysis.

Bumping either version constant here is a schema-breaking change: the
analysis-side guard (``aiworld/analysis/provenance.py::require_completed_run``)
refuses to treat runs logged under an incompatible version as comparable, so a
future format change can't silently get pooled with older, differently
shaped data.
"""

from __future__ import annotations

import subprocess

# The action grammar/interface a run's policy spoke. "v1" is the free-text
# grammar every primary/open-world condition uses today
# (aiworld/components/llm_policy.py). "v2" is the strict, opaque-menu
# interface (aiworld/actions.py) used only at probe/branch decision points --
# it never replaces v1 as the primary condition's interface.
ACTION_API_VERSION = "v1"

# The tick/behavioral-feature schema version analysis code assumes. Bump this
# when TickRecord or FEATURE_NAMES changes shape in a way that makes old and
# new runs non-comparable -- see behavioral_metrics.py's own "LAYOUT NOTE".
ANALYSIS_SCHEMA_VERSION = 1


def git_revision() -> str | None:
    """Best-effort short-circuits-to-None git commit hash for the working tree.

    Never raises: provenance should degrade gracefully outside a git
    checkout, without git installed, or under any other transient failure,
    rather than crashing a run over a cosmetic metadata field.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
