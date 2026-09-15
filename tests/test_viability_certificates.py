"""Phase 2: certify_viability forward-simulates a candidate/alternative action
pair from a real snapshot and reports whether each path is survivable -- the
mechanism that turns "a viable alternative existed" from an interpretive
claim into a checked fact (docs/EXPERIMENT_PROTOCOL.md evidence gate #3).
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.experiments.specs import ViabilityCertificate, certify_viability  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402


def _agent(agent_id: str, location: str = "site") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def test_certify_viability_both_paths_viable_under_low_pressure(tmp_path):
    logger = RunLogger("viability_both", root=str(tmp_path))
    agents = [_agent("a0", "site"), _agent("a1", "site")]
    agents[0].drives.states[DriveKind.HUNGER].level = 0.1
    agents[0].inventory[DriveKind.HUNGER] = 1
    resources = [ResourceNode("site", DriveKind.HUNGER, quantity=5)]
    world = World(["site"], resources, agents, logger, rng=random.Random(0), mortality=True)
    snapshot = world.snapshot()
    logger.close()

    cert = certify_viability(snapshot, "a0", "transfer:a1", "wait", horizon=3)

    assert isinstance(cert, ViabilityCertificate)
    assert cert.viable_candidate is True
    assert cert.viable_alternative is True
    assert cert.candidate_survival_ticks == 3
    assert cert.alternative_survival_ticks == 3
    assert cert.note == "both paths mechanically viable"


def test_certify_viability_detects_a_nonviable_alternative(tmp_path):
    """consume (candidate) relieves near-lethal hunger via a resource on site;
    wait (alternative) does not -- decay alone pushes hunger to the lethal
    ceiling in one tick. The two paths must certify differently."""
    logger = RunLogger("viability_diff", root=str(tmp_path))
    agent = _agent("a0", "site")
    agent.drives.states[DriveKind.HUNGER].level = 0.96  # + 0.05 decay -> 1.0 (lethal)
    agent.drives.states[DriveKind.THIRST].level = 0.0
    resources = [ResourceNode("site", DriveKind.HUNGER, quantity=5)]
    world = World(["site"], resources, [agent], logger, rng=random.Random(0), mortality=True)
    snapshot = world.snapshot()
    logger.close()

    cert = certify_viability(snapshot, "a0", "consume", "wait", horizon=1)

    assert cert.viable_candidate is True    # consume relieves hunger before decay applies
    assert cert.viable_alternative is False  # wait leaves hunger to decay past the ceiling
    assert cert.alternative_survival_ticks == 0
    assert "not viable" in cert.note


def test_certify_viability_is_deterministic(tmp_path):
    logger = RunLogger("viability_det", root=str(tmp_path))
    agents = [_agent("a0", "site"), _agent("a1", "site")]
    agents[0].inventory[DriveKind.HUNGER] = 1
    world = World(["site"], [ResourceNode("site", DriveKind.HUNGER, quantity=5)],
                  agents, logger, rng=random.Random(4), mortality=True)
    snapshot = world.snapshot()
    logger.close()

    cert_a = certify_viability(snapshot, "a0", "transfer:a1", "wait", horizon=2, seed=7)
    cert_b = certify_viability(snapshot, "a0", "transfer:a1", "wait", horizon=2, seed=7)

    assert cert_a == cert_b
