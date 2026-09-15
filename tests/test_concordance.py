"""Phase 2: the free-text <-> menu interface concordance harness.

Generic over policy callables -- tested here with fake/scripted policies,
with no dependency on Phase 3's concrete menu implementation.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem  # noqa: E402
from aiworld.experiments.concordance import compare_interfaces  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, World  # noqa: E402


def _agent(agent_id: str, location: str = "site") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def _build_snapshot(tmp_path):
    logger = RunLogger("concordance_source", root=str(tmp_path))
    world = World(["site"], [], [_agent("a0")], logger, mortality=False)
    snapshot = world.snapshot()
    logger.close()
    return snapshot


def test_compare_interfaces_reports_full_agreement(tmp_path):
    snapshot = _build_snapshot(tmp_path)

    result = compare_interfaces(
        snapshot, "a0", n_samples=3,
        free_text_policy=lambda agent, perceived, world: "wait",
        menu_policy=lambda agent, world: "wait",
        logger_root=str(tmp_path / "logs"))

    assert result.n_samples == 3
    assert result.agreement_rate == 1.0
    assert result.disagreements == []


def test_compare_interfaces_reports_partial_agreement(tmp_path):
    snapshot = _build_snapshot(tmp_path)
    calls = {"n": 0}

    def menu_policy(agent, world):
        calls["n"] += 1
        return "wait" if calls["n"] % 2 else "move:elsewhere"

    result = compare_interfaces(
        snapshot, "a0", n_samples=4,
        free_text_policy=lambda agent, perceived, world: "wait",
        menu_policy=menu_policy,
        logger_root=str(tmp_path / "logs"))

    assert result.n_samples == 4
    assert result.agreement_rate == 0.5
    assert len(result.disagreements) == 2
    assert all(d["free_text_action"] == "wait" for d in result.disagreements)
    assert all(d["menu_action"] == "move:elsewhere" for d in result.disagreements)


def test_compare_interfaces_compares_action_kind_not_exact_target(tmp_path):
    """A menu's opaque target ID and free text's literal agent_id should
    count as agreement when both are the SAME kind of action (e.g. both
    'talk'), since exact IDs will never literally match across interfaces."""
    snapshot = _build_snapshot(tmp_path)

    result = compare_interfaces(
        snapshot, "a0", n_samples=1,
        free_text_policy=lambda agent, perceived, world: "talk:a1",
        menu_policy=lambda agent, world: "talk:Q4",
        logger_root=str(tmp_path / "logs"))

    assert result.agreement_rate == 1.0
