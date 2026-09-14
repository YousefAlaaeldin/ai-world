"""Tests for Tranche 2 (docs/REDESIGN_PLAN.md): the opaque vocabulary arm and
factual (de-narrativized) memory. Pure Python, no LLM.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem, DriveKind, DriveSpec  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.sim.world import World, Agent, ResourceNode  # noqa: E402
from aiworld.vocabulary import BIOLOGICAL_TERMS  # noqa: E402

# Words the redesign review flagged as narrative loading -- must not appear
# anywhere in a factual-style memory/outcome string.
NARRATIVE_TERMS = ("nearly died", "rescued", "salient", "important",
                   "approached you and described", "who died")


# ---- opaque vocabulary arm ----

def test_opaque_state_is_fully_symbolic():
    ds = DriveSystem(vocabulary="opaque")
    ds.states[DriveKind.HUNGER].level = 0.42
    text = ds.as_prompt_block()
    assert "S1: 0.42" in text
    assert text.startswith("State —") or text.startswith("State —")
    # No biological OR neutral-semantic words leak through.
    for banned in list(BIOLOGICAL_TERMS) + ["deficit", "isolation", "overwhelming",
                                            "mild", "strong", "not noticeable"]:
        assert banned not in text.lower()


def test_opaque_resource_labels_are_symbolic(tmp_path):
    a = Agent("a0", "loc", DriveSystem(vocabulary="opaque"), SelfModelHistory("a0"))
    res = [ResourceNode("loc", DriveKind.HUNGER, quantity=5)]
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["loc"], res, [a], logger, vocabulary="opaque")
    obs = world.observations_for(a)
    logger.close()
    texts = " ".join(o.text for o in obs).lower()
    assert world.resource_label(DriveKind.HUNGER).lower() in texts
    for banned in BIOLOGICAL_TERMS:
        assert banned not in texts


def test_opaque_and_neutral_are_distinct_arms():
    ds_neutral = DriveSystem(vocabulary="neutral")
    ds_opaque = DriveSystem(vocabulary="opaque")
    ds_neutral.states[DriveKind.HUNGER].level = 0.9
    ds_opaque.states[DriveKind.HUNGER].level = 0.9
    assert ds_neutral.as_prompt_block() != ds_opaque.as_prompt_block()
    assert "energy deficit" in ds_neutral.as_prompt_block()
    assert "S1" in ds_opaque.as_prompt_block()


# ---- factual memory style ----

def test_factual_consume_reports_state_delta(tmp_path):
    a = Agent("a0", "loc", DriveSystem(vocabulary="neutral"), SelfModelHistory("a0"))
    a.drives.states[DriveKind.HUNGER].level = 0.9
    res = [ResourceNode("loc", DriveKind.HUNGER, quantity=5)]
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["loc"], res, [a], logger, vocabulary="neutral", memory_style="factual")
    outcome = world.resolve_action(a, "consume")
    logger.close()
    assert "0.90" in outcome and "->" in outcome
    for banned in NARRATIVE_TERMS:
        assert banned not in outcome


def test_factual_move_reports_origin_and_destination(tmp_path):
    a = Agent("a0", "north", DriveSystem(), SelfModelHistory("a0"))
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["north", "center"], [], [a], logger,
                 adjacency={"north": {"center"}, "center": {"north"}},
                 memory_style="factual")
    outcome = world.resolve_action(a, "move:center")
    logger.close()
    assert outcome == "moved from north to center"


def test_factual_depletion_has_no_narrative_framing(tmp_path):
    a = Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0"))
    res = [ResourceNode("loc", DriveKind.HUNGER, quantity=1)]
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["loc"], res, [a], logger, memory_style="factual")
    world.resolve_action(a, "consume")  # depletes the only unit
    logger.close()
    text = " ".join(a.long_term_memory)
    assert "depleted" in text
    for banned in NARRATIVE_TERMS:
        assert banned not in text


def test_factual_crisis_memory_has_no_distress_framing(tmp_path):
    specs = {DriveKind.HUNGER: DriveSpec(DriveKind.HUNGER, decay_per_tick=0.9,
                                         replenish_amount=1.0, lethal=True)}
    agent = Agent("a0", "loc", DriveSystem(specs=specs), SelfModelHistory("a0"))
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["loc"], [], [agent], logger, mortality=True, memory_style="factual")
    world.run(lambda a, p, w: "wait", ticks=1, reflect_every=1000, reflect_fn=None)
    logger.close()
    text = " ".join(agent.long_term_memory)
    assert "0.80" in text or ">=" in text   # threshold-cross record present
    for banned in NARRATIVE_TERMS:
        assert banned not in text


def test_factual_attack_and_take_avoid_narrative_framing(tmp_path):
    a, b = (Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0")),
            Agent("a1", "loc", DriveSystem(), SelfModelHistory("a1")))
    b.inventory[DriveKind.THIRST] = 1
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["loc"], [], [a, b], logger, memory_style="factual")
    world.resolve_action(a, "take:a1")
    world.resolve_action(a, "attack:a1")
    logger.close()
    text = " ".join(b.recent_memories + b.long_term_memory)
    for banned in NARRATIVE_TERMS:
        assert banned not in text
    assert "lost 1" in text or "taken by" in text
    assert "attacked by" in text


def test_factual_identity_channel_is_an_explicit_positive_control(tmp_path):
    a, b = (Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0")),
            Agent("a1", "loc", DriveSystem(), SelfModelHistory("a1")))
    b.history.append(b.history.current.revise(who_i_am="a wanderer", tick=1))
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["loc"], [], [a, b], logger, memory_style="factual",
                  communication_mode="self_model")
    outcome = world.resolve_action(a, "talk:a1")
    logger.close()
    # Content transmission is now an explicit treatment, rather than an
    # automatic side effect of social contact.
    assert "a wanderer" in outcome
    assert "approached you and described" not in " ".join(b.recent_memories)
    assert any("message from a0" in m for m in b.recent_memories)


def test_narrative_style_is_unchanged_default(tmp_path):
    # Byte-identical to pre-Tranche-2 behavior when memory_style is left at default.
    a = Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0"))
    res = [ResourceNode("loc", DriveKind.HUNGER, quantity=1)]
    logger = RunLogger("t", root=str(tmp_path))
    world = World(["loc"], res, [a], logger)   # no memory_style kwarg -> default
    outcome = world.resolve_action(a, "consume")
    logger.close()
    assert outcome == "consumed energy resource at loc"
    assert any("ran out while you were there" in m for m in a.long_term_memory)
