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


def test_salience_reweights_never_hides():
    obs = [
        Observation("a pile of fruit", related_drive=DriveKind.HUNGER),
        Observation("an empty clearing", related_drive=None),
    ]
    biases = {DriveKind.HUNGER: 5.0}
    ranked = rank_observations(obs, biases)
    assert len(ranked) == 2  # nothing dropped
