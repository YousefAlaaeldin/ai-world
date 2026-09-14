"""Tests for the pure-Python core: drives, mortality, self-model, salience.

None of these need an LLM. They lock in the mechanics before any model is wired up.
Run with:  python -m pytest tests/ -v   (from the project root)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem, DriveKind, DriveSpec  # noqa: E402
from aiworld.self_model import SelfModel, SelfModelHistory, SHARED_DIRECTIVE  # noqa: E402
from aiworld.salience import Observation, rank_observations  # noqa: E402
from aiworld.sim.world import (  # noqa: E402
    Agent, World, ResourceNode, agent_recent_memories, remember_long_term,
    LONG_TERM_MEMORY_CAP)
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.config import initial_locations, adjacency_from_config  # noqa: E402
from aiworld.vocabulary import BIOLOGICAL_TERMS  # noqa: E402
from aiworld.salience import render_perceived  # noqa: E402
from aiworld.components.llm_policy import (  # noqa: E402
    _build_action_prompt, _action_system_prompt)
import random  # noqa: E402


# ---- drives & mortality ----

def test_drives_decay_upward():
    ds = DriveSystem()
    start = ds.states[DriveKind.HUNGER].level
    ds.step(tick=1)
    assert ds.states[DriveKind.HUNGER].level > start


def test_satisfy_reduces_drive():
    ds = DriveSystem()
    for _ in range(10):
        ds.step(tick=_)
    high = ds.states[DriveKind.HUNGER].level
    ds.satisfy(DriveKind.HUNGER)
    assert ds.states[DriveKind.HUNGER].level < high


def test_lethal_drive_kills_agent():
    ds = DriveSystem()
    # Force thirst to the ceiling.
    for t in range(100):
        ds.step(tick=t)
        if not ds.alive:
            break
    assert not ds.alive
    assert ds.cause_of_death in {"hunger", "thirst"}
    assert ds.died_at_tick is not None


def test_immortal_mode_never_kills():
    ds = DriveSystem()
    for t in range(200):
        ds.step(tick=t, mortality=False)
    assert ds.alive
    assert ds.states[DriveKind.HUNGER].level == 1.0  # still hits crisis level...
    assert ds.states[DriveKind.THIRST].level == 1.0  # ...just never dies from it


def test_in_crisis_tracks_lethal_drives_only():
    ds = DriveSystem()
    assert not ds.in_crisis()
    ds.states[DriveKind.AFFILIATION].level = 1.0  # non-lethal maxed
    assert not ds.in_crisis()                     # doesn't count
    ds.states[DriveKind.HUNGER].level = 0.85       # lethal, in "overwhelming" zone
    assert ds.in_crisis()


def test_affiliation_is_not_lethal():
    spec = {DriveKind.AFFILIATION: DriveSpec(
        DriveKind.AFFILIATION, decay_per_tick=0.5, replenish_amount=0.4, lethal=False)}
    ds = DriveSystem(specs=spec)
    for t in range(10):
        ds.step(tick=t)
    assert ds.states[DriveKind.AFFILIATION].level == 1.0  # maxed
    assert ds.alive  # but not dead


def test_salience_weight_is_nonlinear():
    ds = DriveSystem()
    ds.states[DriveKind.HUNGER].level = 0.5
    w_half = ds.states[DriveKind.HUNGER].salience_weight()
    ds.states[DriveKind.HUNGER].level = 1.0
    w_full = ds.states[DriveKind.HUNGER].salience_weight()
    assert w_full > 2 * w_half  # quadratic: crisis dominates


# ---- self-model ----

def test_all_agents_start_identical():
    a = SelfModel.initial()
    b = SelfModel.initial()
    assert a.embedding_text() == b.embedding_text()
    assert a.what_i_want_now == SHARED_DIRECTIVE
    assert a.who_i_am == "" and a.what_matters == ""


def test_revise_preserves_history_and_bumps_version():
    hist = SelfModelHistory("agent_0")
    assert hist.current.version == 0
    revised = hist.current.revise(
        who_i_am="a gatherer", tick=5, triggering_event="found a cache of food")
    hist.append(revised)
    assert hist.current.version == 1
    assert hist.current.who_i_am == "a gatherer"
    assert len(hist.versions) == 2          # original preserved
    assert hist.versions[0].who_i_am == ""   # untouched


def test_self_model_roundtrip():
    m = SelfModel(who_i_am="x", what_matters="y", what_i_want_now="z", version=3)
    assert SelfModel.from_dict(m.to_dict()) == m


# ---- salience ----

def test_hungry_agent_ranks_food_first():
    obs = [
        Observation("a flowing stream", related_drive=DriveKind.THIRST),
        Observation("a pile of fruit", related_drive=DriveKind.HUNGER),
        Observation("an empty clearing", related_drive=None),
    ]
    # Very hungry, not thirsty.
    biases = {DriveKind.HUNGER: 1.0, DriveKind.THIRST: 0.0}
    ranked = rank_observations(obs, biases)
    assert ranked[0][0].text == "a pile of fruit"


def test_recent_memories_accumulate_across_ticks():
    agent = Agent("agent_0", "north", DriveSystem(), SelfModelHistory("agent_0"))
    agent.recent_memories.append("tick 1: did 'wait' -> waited")
    agent.recent_memories.append("tick 2: did 'consume' -> consumed hunger resource")
    text = agent_recent_memories(agent)
    assert "tick 1" in text and "tick 2" in text


def test_recent_memories_drain_after_read():
    agent = Agent("agent_0", "north", DriveSystem(), SelfModelHistory("agent_0"))
    agent.recent_memories.append("tick 1: did 'wait' -> waited")
    agent_recent_memories(agent)
    assert agent.recent_memories == []  # buffer cleared, so next reflection
                                        # only sees ticks since THIS reflection


def test_long_term_memory_persists_across_reads():
    agent = Agent("agent_0", "north", DriveSystem(), SelfModelHistory("agent_0"))
    remember_long_term(agent, "tick 1: nearly died from thirst")
    agent_recent_memories(agent)   # drains recent_memories, NOT long_term_memory
    agent_recent_memories(agent)   # a second read still sees it
    assert agent.long_term_memory == ["tick 1: nearly died from thirst"]


def test_long_term_memory_caps_with_fifo_eviction():
    agent = Agent("agent_0", "north", DriveSystem(), SelfModelHistory("agent_0"))
    for i in range(LONG_TERM_MEMORY_CAP + 5):
        remember_long_term(agent, f"event {i}")
    assert len(agent.long_term_memory) == LONG_TERM_MEMORY_CAP
    assert agent.long_term_memory[0] == "event 5"   # oldest 5 evicted
    assert agent.long_term_memory[-1] == f"event {LONG_TERM_MEMORY_CAP + 4}"


def test_agent_recent_memories_includes_long_term_section():
    agent = Agent("agent_0", "north", DriveSystem(), SelfModelHistory("agent_0"))
    remember_long_term(agent, "tick 1: nearly died from thirst")
    text = agent_recent_memories(agent)
    assert "Further back, you remember" in text
    assert "nearly died from thirst" in text


def test_agent_recent_memories_omits_long_term_section_when_empty():
    agent = Agent("agent_0", "north", DriveSystem(), SelfModelHistory("agent_0"))
    text = agent_recent_memories(agent)
    assert "Further back" not in text


def test_recent_memories_empty_has_sane_fallback():
    agent = Agent("agent_0", "north", DriveSystem(), SelfModelHistory("agent_0"))
    assert agent_recent_memories(agent) == "(nothing notable happened)"


def test_consume_picks_most_urgent_matching_resource(tmp_path):
    # Two resource types co-located -- the exact shape that starved agents
    # before the fix (consume always grabbed the first-listed resource).
    resources = [ResourceNode("loc", DriveKind.HUNGER, quantity=10),
                ResourceNode("loc", DriveKind.THIRST, quantity=10)]
    agent = Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0"))
    agent.drives.states[DriveKind.THIRST].level = 0.9
    agent.drives.states[DriveKind.HUNGER].level = 0.1
    logger = RunLogger("test_consume", root=str(tmp_path))
    world = World(["loc"], resources, [agent], logger)
    outcome = world.resolve_action(agent, "consume")
    logger.close()
    assert "thirst" in outcome


def test_talk_exchanges_self_model_content(tmp_path):
    a = Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0"))
    b = Agent("a1", "loc", DriveSystem(), SelfModelHistory("a1"))
    b.history.append(b.history.current.revise(who_i_am="a wanderer", tick=1))
    logger = RunLogger("test_talk", root=str(tmp_path))
    world = World(["loc"], [], [a, b], logger)

    outcome = world.resolve_action(a, "talk:a1")
    logger.close()

    assert "a wanderer" in outcome                                # a learns b's identity
    assert any("a0 approached you" in m for m in b.recent_memories)  # b was notified
    assert not any("a wanderer" in m for m in b.recent_memories)  # b isn't told its own identity


def test_talk_to_absent_agent_fails_safely(tmp_path):
    a = Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0"))
    logger = RunLogger("test_talk_absent", root=str(tmp_path))
    world = World(["loc"], [], [a], logger)

    before = a.drives.states[DriveKind.AFFILIATION].level
    outcome = world.resolve_action(a, "talk:ghost")
    logger.close()

    assert "not here" in outcome
    assert a.drives.states[DriveKind.AFFILIATION].level == before  # no false satisfaction


def test_crisis_triggers_immediate_reflection(tmp_path):
    # decay_per_tick=0.9 means hunger crosses the 0.8 crisis line on tick 1,
    # well before reflect_every (set huge) would ever fire on schedule.
    specs = {DriveKind.HUNGER: DriveSpec(DriveKind.HUNGER, decay_per_tick=0.9,
                                         replenish_amount=1.0, lethal=True)}
    agent = Agent("a0", "loc", DriveSystem(specs=specs), SelfModelHistory("a0"))
    logger = RunLogger("test_crisis_reflect", root=str(tmp_path))
    world = World(["loc"], [], [agent], logger, mortality=True)

    calls = []

    def fake_policy(agent, perceived, world):
        return "wait"

    def fake_reflect(agent, perceived, outcome, tick, reason="scheduled"):
        calls.append((tick, reason))

    world.run(fake_policy, ticks=1, reflect_every=1000, reflect_fn=fake_reflect)
    logger.close()

    assert calls == [(1, "crisis")]
    assert any("nearly died from hunger" in m for m in agent.long_term_memory)


def test_resource_depletion_persists_to_long_term_memory(tmp_path):
    resources = [ResourceNode("loc", DriveKind.HUNGER, quantity=1)]
    agent = Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0"))
    logger = RunLogger("test_depletion_ltm", root=str(tmp_path))
    world = World(["loc"], resources, [agent], logger)
    world.resolve_action(agent, "consume")  # depletes the only unit -> quantity 0
    logger.close()
    assert any("ran out" in m for m in agent.long_term_memory)


def test_talk_persists_to_both_parties_long_term_memory(tmp_path):
    a = Agent("a0", "loc", DriveSystem(), SelfModelHistory("a0"))
    b = Agent("a1", "loc", DriveSystem(), SelfModelHistory("a1"))
    logger = RunLogger("test_talk_ltm", root=str(tmp_path))
    world = World(["loc"], [], [a, b], logger)
    world.resolve_action(a, "talk:a1")
    logger.close()
    assert any("talked with a1" in m for m in a.long_term_memory)
    assert any("a0 approached" in m for m in b.long_term_memory)


def test_adjacency_from_config_uses_explicit_edges():
    cfg = {"world": {"locations": ["a", "b", "c"], "edges": [["a", "b"]]}}
    adjacency = adjacency_from_config(cfg)
    assert adjacency["a"] == {"b"}
    assert adjacency["b"] == {"a"}
    assert adjacency["c"] == set()  # isolated -- no edge touches it


def test_adjacency_from_config_falls_back_to_fully_connected():
    cfg = {"world": {"locations": ["a", "b", "c"]}}  # no edges key
    adjacency = adjacency_from_config(cfg)
    assert adjacency["a"] == {"b", "c"}


def test_move_rejected_when_not_adjacent(tmp_path):
    logger = RunLogger("test_move", root=str(tmp_path))
    agent = Agent("a0", "hub", DriveSystem(), SelfModelHistory("a0"))
    world = World(["hub", "spoke", "far"], [], [agent], logger,
                 adjacency={"hub": {"spoke"}, "spoke": {"hub"}, "far": set()})
    outcome = world.resolve_action(agent, "move:far")  # not adjacent to hub
    logger.close()
    assert "unreachable" in outcome
    assert agent.location == "hub"  # unchanged


def test_next_hop_towards_multi_hop(tmp_path):
    agent = Agent("a0", "north", DriveSystem(), SelfModelHistory("a0"))
    adjacency = {"north": {"center"}, "south": {"center"},
                "center": {"north", "south"}}
    logger = RunLogger("test_hop", root=str(tmp_path))
    world = World(["north", "south", "center"], [], [agent],
                 logger, adjacency=adjacency)
    assert world.next_hop_towards("north", "south") == "center"  # 2-hop path
    assert world.next_hop_towards("north", "center") == "center"  # direct
    assert world.next_hop_towards("north", "north") is None       # already there
    logger.close()


def test_initial_locations_isolates_when_enough_locations():
    rng = random.Random(0)
    locs = initial_locations(rng, ["a", "b", "c", "d", "e", "f"], 6)
    assert len(locs) == 6
    assert len(set(locs)) == 6  # every agent gets a distinct location


def test_initial_locations_allows_collision_when_scarce():
    rng = random.Random(0)
    locs = initial_locations(rng, ["a", "b"], 6)
    assert len(locs) == 6
    assert all(l in ("a", "b") for l in locs)


def test_salience_reweights_never_hides():
    obs = [
        Observation("a pile of fruit", related_drive=DriveKind.HUNGER),
        Observation("an empty clearing", related_drive=None),
    ]
    biases = {DriveKind.HUNGER: 5.0}
    ranked = rank_observations(obs, biases)
    assert len(ranked) == 2  # nothing dropped


# ---- vocabulary (the safety-relevant confound control) ----

def _assemble_action_prompt(vocabulary, tmp_path):
    """Build the FULL model-facing action text (system prompt + assembled prompt)
    for an agent that is co-located with a resource and has non-trivial drive
    levels -- so every render path (state block, perceived resource) contributes."""
    agent = Agent("a0", "loc", DriveSystem(vocabulary=vocabulary),
                  SelfModelHistory("a0"))
    agent.drives.states[DriveKind.HUNGER].level = 0.9
    agent.drives.states[DriveKind.THIRST].level = 0.3
    resources = [ResourceNode("loc", DriveKind.HUNGER, quantity=5),
                 ResourceNode("loc", DriveKind.THIRST, quantity=5)]
    logger = RunLogger("test_vocab", root=str(tmp_path))
    world = World(["loc"], resources, [agent], logger, vocabulary=vocabulary)
    perceived = render_perceived(world.observations_for(agent),
                                 agent.drives.salience_biases(), top_k=6)
    prompt = _build_action_prompt(agent, perceived, world)
    logger.close()
    return _action_system_prompt(vocabulary) + "\n" + prompt


def test_neutral_vocabulary_leaks_no_biological_terms(tmp_path):
    # The whole point of the neutral arm: nothing the model reads should invoke
    # its biological survival priors. Lock that property on the ASSEMBLED prompt,
    # not on any one function, so it can't regress via a new render path.
    text = _assemble_action_prompt("neutral", tmp_path).lower()
    leaked = [t for t in BIOLOGICAL_TERMS if t in text]
    assert leaked == [], f"neutral prompt leaked biological terms: {leaked}"


def test_biological_vocabulary_restores_concrete_terms(tmp_path):
    # The contrast arm must actually differ -- otherwise the confound control is a
    # no-op and comparing the two arms proves nothing.
    text = _assemble_action_prompt("biological", tmp_path).lower()
    assert "food" in text and ("hunger" in text or "bodily" in text)


def test_vocabulary_arms_differ_only_in_nouns():
    # Structural identity: the two arms must produce the SAME prompt shape, so any
    # behavioral difference between them is attributable to the words, not the
    # layout. Substituting neutral nouns -> biological nouns must recover the
    # biological rendering exactly.
    levels = {DriveKind.HUNGER: 0.9, DriveKind.THIRST: 0.3, DriveKind.AFFILIATION: 0.6}
    neutral = DriveSystem(vocabulary="neutral")
    biological = DriveSystem(vocabulary="biological")
    for k, v in levels.items():
        neutral.states[k].level = v
        biological.states[k].level = v
    subs = [("energy deficit", "hunger"), ("maintenance deficit", "thirst"),
            ("isolation", "affiliation"), ("Internal state", "Bodily state")]
    rendered = neutral.as_prompt_block()
    for neutral_term, bio_term in subs:
        rendered = rendered.replace(neutral_term, bio_term)
    assert rendered == biological.as_prompt_block()
