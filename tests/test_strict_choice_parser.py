"""Phase 3: the strict-choice parser must never substring-extract an action
from prose, and invalid output must never become an intentional 'wait'.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.actions import (  # noqa: E402
    InvalidChoiceError,
    StrictChoiceRecord,
    make_strict_choice_policy,
    parse_strict_choice,
)
from aiworld.drives import DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, World  # noqa: E402


_ID_MAP = {"K7": "attack:agent_1", "Q4": "wait", "M9": "move:south"}


def test_parse_strict_choice_accepts_exact_json():
    action, status = parse_strict_choice('{"choice_id":"Q4"}', _ID_MAP)
    assert action == "wait"
    assert status == "ok"


def test_parse_strict_choice_tolerates_surrounding_whitespace():
    action, status = parse_strict_choice('  {"choice_id":"M9"}  \n', _ID_MAP)
    assert action == "move:south"
    assert status == "ok"


def test_parse_strict_choice_never_executes_attack_from_prose_mentioning_it():
    """The literal acceptance test from the reviewed plan: text that merely
    MENTIONS an action string must never parse as that action."""
    action, status = parse_strict_choice("I will not attack:agent_1", _ID_MAP)
    assert action is None
    assert action != "attack:agent_1"
    assert status == "invalid_output"


def test_parse_strict_choice_rejects_prose_around_valid_json():
    action, status = parse_strict_choice(
        'Sure, here is my choice: {"choice_id":"K7"}', _ID_MAP)
    assert action is None
    assert status == "invalid_output"


def test_parse_strict_choice_rejects_unknown_id():
    action, status = parse_strict_choice('{"choice_id":"ZZ"}', _ID_MAP)
    assert action is None
    assert status == "unknown_choice_id"


def test_parse_strict_choice_rejects_empty_completion():
    action, status = parse_strict_choice("", _ID_MAP)
    assert action is None
    assert status == "empty_completion"
    action, status = parse_strict_choice(None, _ID_MAP)
    assert action is None
    assert status == "empty_completion"


class _FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _agent(agent_id: str = "a0", location: str = "north") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def test_make_strict_choice_policy_resolves_a_valid_choice(tmp_path):
    logger = RunLogger("strict1", root=str(tmp_path))
    world = World(["north", "south"], [], [_agent()], logger, mortality=False,
                  adjacency={"north": {"south"}, "south": {"north"}})
    client = _FakeClient('{"choice_id":"AA"}')  # placeholder; real id assigned below

    records: list[StrictChoiceRecord] = []
    policy = make_strict_choice_policy(client, rng=random.Random(0),
                                       on_decision=records.append)
    # Discover the id actually assigned to "wait" this decision, then have
    # the fake client answer with it -- exercises the real id_map, not a
    # guessed literal.
    from aiworld.actions import assign_opaque_ids, legal_action_menu
    menu = legal_action_menu(world, world.agents[0])
    id_map = assign_opaque_ids(menu, random.Random(0))
    wait_id = next(cid for cid, action in id_map.items() if action == "wait")
    client.response = f'{{"choice_id":"{wait_id}"}}'

    action = policy(world.agents[0], world)
    logger.close()

    assert action == "wait"
    assert len(records) == 1
    assert records[0].parse_status == "ok"
    assert records[0].parsed_action == "wait"


def test_make_strict_choice_policy_never_falls_back_to_wait_on_invalid_output(tmp_path):
    logger = RunLogger("strict2", root=str(tmp_path))
    world = World(["north"], [], [_agent()], logger, mortality=False)
    client = _FakeClient("I decline to choose.")

    records: list[StrictChoiceRecord] = []
    policy = make_strict_choice_policy(client, rng=random.Random(0),
                                       on_decision=records.append)

    with pytest.raises(InvalidChoiceError):
        policy(world.agents[0], world)
    logger.close()

    # The decision is still recorded (for the run's provenance) even though
    # it raised -- an aborted branch keeps its audit trail.
    assert len(records) == 1
    assert records[0].parse_status == "invalid_output"
    assert records[0].parsed_action is None
