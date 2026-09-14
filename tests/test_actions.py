"""Tests for the Tranche 4 richer action space: gather / transfer / build / take /
attack, plus consume-from-store. Pure Python, no LLM.

Two of these are load-bearing for the research design, per review:
  - mortality must stay REACHABLE (consume-from-store must not make death impossible)
  - multiple strategies must be VIABLE (a hoarder and a hand-to-mouth agent both
    survive a fixed scenario) -- otherwise a hoarding cluster is ordinary adaptation,
    not emergent purpose.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem, DriveKind, DriveSpec  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.sim.world import World, Agent, ResourceNode, INVENTORY_CAP  # noqa: E402
from aiworld.components.llm_policy import parse_action  # noqa: E402


def _agent(aid="a0", loc="loc", specs=None):
    return Agent(aid, loc, DriveSystem(specs=specs), SelfModelHistory(aid))


def _world(tmp_path, agents, resources=None, locations=None, mortality=True):
    locations = locations or ["loc"]
    logger = RunLogger("t", root=str(tmp_path))
    return World(locations, resources or [], agents, logger, mortality=mortality), logger


# ---- gather / store ----

def test_gather_stores_a_unit(tmp_path):
    a = _agent()
    res = [ResourceNode("loc", DriveKind.HUNGER, quantity=5)]
    world, logger = _world(tmp_path, [a], res)
    out = world.resolve_action(a, "gather")
    logger.close()
    assert a.inventory[DriveKind.HUNGER] == 1
    assert res[0].quantity == 4
    assert "gathered" in out


def test_gather_respects_cap(tmp_path):
    a = _agent()
    a.inventory[DriveKind.HUNGER] = INVENTORY_CAP
    res = [ResourceNode("loc", DriveKind.HUNGER, quantity=5)]
    world, logger = _world(tmp_path, [a], res)
    out = world.resolve_action(a, "gather")
    logger.close()
    assert a.inventory[DriveKind.HUNGER] == INVENTORY_CAP   # unchanged
    assert res[0].quantity == 5 and "full" in out


# ---- consume fallbacks ----

def test_consume_falls_back_to_store_then_public(tmp_path):
    # No node here; a stored unit must be usable, and reduce the drive.
    a = _agent()
    a.inventory[DriveKind.THIRST] = 1
    a.drives.states[DriveKind.THIRST].level = 0.9
    world, logger = _world(tmp_path, [a])
    out = world.resolve_action(a, "consume")
    logger.close()
    assert a.inventory[DriveKind.THIRST] == 0
    assert a.drives.states[DriveKind.THIRST].level < 0.9
    assert "stored" in out


# ---- transfer (give) ----

def test_transfer_moves_unit_and_notifies_recipient(tmp_path):
    a, b = _agent("a0"), _agent("a1")
    a.inventory[DriveKind.HUNGER] = 2
    world, logger = _world(tmp_path, [a, b])
    out = world.resolve_action(a, "transfer:a1")
    logger.close()
    assert a.inventory[DriveKind.HUNGER] == 1
    assert b.inventory[DriveKind.HUNGER] == 1
    assert "transferred" in out
    assert any("transferred 1" in m for m in b.recent_memories)


def test_transfer_with_empty_store_fails(tmp_path):
    a, b = _agent("a0"), _agent("a1")
    world, logger = _world(tmp_path, [a, b])
    out = world.resolve_action(a, "transfer:a1")
    logger.close()
    assert "empty" in out and not b.inventory


# ---- build (public good) ----

def test_build_creates_shared_stock_others_can_consume(tmp_path):
    a, b = _agent("a0"), _agent("a1")
    a.inventory[DriveKind.HUNGER] = 1
    b.drives.states[DriveKind.HUNGER].level = 0.9
    world, logger = _world(tmp_path, [a, b])
    world.resolve_action(a, "build")
    assert world.public_stock["loc"][DriveKind.HUNGER] == 1
    out = world.resolve_action(b, "consume")   # b has no node/store; uses public
    logger.close()
    assert world.public_stock["loc"][DriveKind.HUNGER] == 0
    assert b.drives.states[DriveKind.HUNGER].level < 0.9
    assert "shared stock" in out


# ---- take (steal) ----

def test_take_steals_from_target(tmp_path):
    a, b = _agent("a0"), _agent("a1")
    b.inventory[DriveKind.THIRST] = 2
    world, logger = _world(tmp_path, [a, b])
    out = world.resolve_action(a, "take:a1")
    logger.close()
    assert a.inventory[DriveKind.THIRST] == 1
    assert b.inventory[DriveKind.THIRST] == 1
    assert "took 1" in out
    assert any("took 1" in m for m in b.long_term_memory)


# ---- attack (fight / kill) ----

def test_attack_raises_target_drive(tmp_path):
    a, b = _agent("a0"), _agent("a1")
    b.drives.states[DriveKind.HUNGER].level = 0.2
    world, logger = _world(tmp_path, [a, b])
    world.resolve_action(a, "attack:a1")
    logger.close()
    assert b.drives.states[DriveKind.HUNGER].level > 0.2
    assert b.drives.alive
    assert any("attacked you" in m for m in b.long_term_memory)


def test_attack_can_kill(tmp_path):
    a, b = _agent("a0"), _agent("a1")
    b.drives.states[DriveKind.HUNGER].level = 0.8   # + ATTACK_DAMAGE crosses lethal
    world, logger = _world(tmp_path, [a, b], mortality=True)
    out = world.resolve_action(a, "attack:a1")
    logger.close()
    assert not b.drives.alive
    assert b.drives.cause_of_death == "killed"
    assert "who died" in out


def test_directed_action_requires_present_living_target(tmp_path):
    a = _agent("a0")
    world, logger = _world(tmp_path, [a])
    # No agent 'ghost' here -> parse_action demotes to wait.
    assert parse_action("attack:ghost", world, a) == "wait"
    assert parse_action("take:ghost", world, a) == "wait"
    assert parse_action("transfer:ghost", world, a) == "wait"
    logger.close()


# ---- viability & mortality (the load-bearing checks) ----

def test_mortality_still_reachable_with_inventory(tmp_path):
    # Empty store, no resource here: consume-from-store must NOT make death
    # impossible. The agent should still starve.
    a = _agent()
    world, logger = _world(tmp_path, [a], resources=[], mortality=True)
    world.run(lambda ag, p, w: "wait", ticks=40, reflect_every=1000, reflect_fn=None)
    logger.close()
    assert not a.drives.alive


def test_hoarder_and_hand_to_mouth_both_survive(tmp_path):
    # Both strategies must be viable in the same world, or a strategy cluster is
    # just adaptation. Abundant node here; each survives its own way.
    hoarder = _agent("hoarder")
    handtomouth = _agent("handtomouth")
    res = [ResourceNode("loc", DriveKind.HUNGER, quantity=9999),
           ResourceNode("loc", DriveKind.THIRST, quantity=9999)]
    world, logger = _world(tmp_path, [hoarder, handtomouth], res, mortality=True)

    def policy(agent, perceived, w):
        # hand-to-mouth: always consume. hoarder: build a store when comfortable,
        # otherwise consume.
        needy = any(agent.drives.states[k].level >= 0.5
                    for k in (DriveKind.HUNGER, DriveKind.THIRST))
        if agent.agent_id == "handtomouth" or needy:
            return "consume"
        return "gather"

    world.run(policy, ticks=20, reflect_every=1000, reflect_fn=None)
    logger.close()
    assert hoarder.drives.alive
    assert handtomouth.drives.alive
    assert sum(hoarder.inventory.values()) > 0   # it actually built a store
