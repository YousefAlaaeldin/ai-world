"""Phase 7: the 2x2 condition matrix's single-factor validation, and the
non-identity summary control's isolation from identity-framed rendering.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.components.llm_policy import _build_action_prompt  # noqa: E402
from aiworld.components.self_model_component import (  # noqa: E402
    NON_IDENTITY_SUMMARY_PROMPT,
    make_reflect_fn,
    parse_non_identity_summary_diagnostic,
)
from aiworld.config import load_config, validate_paired_configs  # noqa: E402
from aiworld.drives import DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModel, SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, World  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATRIX_DIR = os.path.join(REPO_ROOT, "configs", "experiment_open_world_2x2")


def _load(name: str) -> dict:
    return load_config(os.path.join(MATRIX_DIR, f"{name}.yaml"))


# ---- validate_paired_configs on hand-built dicts ----

def test_validate_paired_configs_allows_difference_only_in_declared_factor():
    a = {"world": {"mortality": True, "vocabulary": "neutral"}}
    b = {"world": {"mortality": False, "vocabulary": "neutral"}}
    validate_paired_configs(a, b, "world.mortality")  # must not raise


def test_validate_paired_configs_raises_on_an_undeclared_difference():
    a = {"world": {"mortality": True, "vocabulary": "neutral"}}
    b = {"world": {"mortality": False, "vocabulary": "biological"}}
    with pytest.raises(ValueError, match="world.vocabulary"):
        validate_paired_configs(a, b, "world.mortality")


def test_validate_paired_configs_subtree_factor_allows_every_nested_key_to_differ():
    a = {"world": {"mortality": True, "resources": [{"quantity": 30}]}}
    b = {"world": {"mortality": True, "resources": [{"quantity": 5}]}}
    # "world.resources" as declared_factor is a leaf-list, not a dict subtree
    # -- flattening stops at non-dict values, so this compares as one leaf.
    validate_paired_configs(a, b, "world.resources")


def test_validate_paired_configs_ignores_experiment_name_and_description():
    a = {"experiment_name": "a", "description": "x", "world": {"mortality": True}}
    b = {"experiment_name": "b", "description": "y", "world": {"mortality": True}}
    validate_paired_configs(a, b, "world.mortality")  # must not raise


def test_validate_paired_configs_raises_when_factor_not_found_in_either():
    a = {"world": {"mortality": True}}
    b = {"world": {"mortality": True}}
    with pytest.raises(ValueError, match="not found"):
        validate_paired_configs(a, b, "world.scarcity_level")


# ---- validate_paired_configs on the real 2x2 configs ----

def test_2x2_mortality_pair_differs_only_in_mortality():
    validate_paired_configs(
        _load("mortality_on_scarcity_low"), _load("mortality_off_scarcity_low"),
        "world.mortality")
    validate_paired_configs(
        _load("mortality_on_scarcity_high"), _load("mortality_off_scarcity_high"),
        "world.mortality")


def test_2x2_scarcity_pair_differs_only_in_resources():
    validate_paired_configs(
        _load("mortality_on_scarcity_low"), _load("mortality_on_scarcity_high"),
        "world.resources")
    validate_paired_configs(
        _load("mortality_off_scarcity_low"), _load("mortality_off_scarcity_high"),
        "world.resources")


def test_2x2_diagonal_pair_differs_in_both_factors():
    """A sanity check on the validator itself: the two cells that differ in
    BOTH mortality and scarcity must be rejected when only one is declared."""
    with pytest.raises(ValueError):
        validate_paired_configs(
            _load("mortality_on_scarcity_low"), _load("mortality_off_scarcity_high"),
            "world.mortality")


# ---- non-identity summary control ----

def test_as_summary_prompt_block_never_says_who_i_am():
    model = SelfModel(who_i_am="", what_matters="", what_i_want_now="events happened")
    block = model.as_summary_prompt_block()
    assert "Who I am" not in block
    assert "What matters" not in block
    assert "Recent summary: events happened" == block


def test_parse_non_identity_summary_keeps_identity_fields_blank():
    previous = SelfModel.initial(directive=None)
    new_model, status = parse_non_identity_summary_diagnostic(
        "Summary: two agents met near the north resource node.",
        previous, tick=5, triggering_event="[scheduled] waited")
    assert status == "ok"
    assert new_model.who_i_am == ""
    assert new_model.what_matters == ""
    assert new_model.what_i_want_now == "two agents met near the north resource node."


def test_parse_non_identity_summary_falls_back_on_malformed_output():
    previous = SelfModel(who_i_am="", what_matters="", what_i_want_now="prior summary")
    new_model, status = parse_non_identity_summary_diagnostic(
        "I refuse to summarize.", previous, tick=5, triggering_event="x")
    assert status == "invalid_format"
    assert new_model.what_i_want_now == "prior summary"  # never wiped


def test_build_action_prompt_with_self_narrative_control_hides_identity_framing(tmp_path):
    logger = RunLogger("control_prompt", root=str(tmp_path))
    try:
        agent = Agent("a0", "north", DriveSystem(),
                      SelfModelHistory("a0", initial_directive=None))
        agent.history.append(agent.history.current.revise(
            who_i_am="should never appear", what_matters="should never appear",
            what_i_want_now="agents gathered resources near north", tick=1,
            triggering_event="x"))
        world = World(["north"], [], [agent], logger, mortality=False)

        prompt = _build_action_prompt(
            agent, "You perceive: nothing.", world, show_self_model=True,
            self_narrative_control=True)
        assert "Who I am" not in prompt
        assert "should never appear" not in prompt
        assert "Recent summary: agents gathered resources near north" in prompt

        default_prompt = _build_action_prompt(
            agent, "You perceive: nothing.", world, show_self_model=True,
            self_narrative_control=False)
        assert "Who I am:" in default_prompt
    finally:
        logger.close()


def test_make_reflect_fn_with_non_identity_control_end_to_end(tmp_path):
    seen_prompts = []

    def fake_model_call(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "Summary: the agent explored a new area."

    hist = SelfModelHistory("a0", initial_directive=None)
    logger = RunLogger("reflect_control", root=str(tmp_path))
    agent = Agent("a0", "north", DriveSystem(), hist)

    reflect = make_reflect_fn(
        model_call=fake_model_call,
        history_by_agent={"a0": hist},
        logger=logger,
        prompt_template=NON_IDENTITY_SUMMARY_PROMPT,
        recent_memories_fn=lambda a: "explored a new area",
        render_self_model_fn=lambda sm: sm.as_summary_prompt_block(),
        parse_fn=parse_non_identity_summary_diagnostic,
    )
    reflect(agent, "perceived", "outcome", tick=5, reason="scheduled")
    logger.close()

    assert "Recent summary:" in seen_prompts[0]
    assert "Who I am" not in seen_prompts[0]
    assert hist.current.who_i_am == ""
    assert hist.current.what_matters == ""
    assert hist.current.what_i_want_now == "the agent explored a new area."
