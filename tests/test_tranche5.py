"""Tranche 5 controls: hidden asymmetry, treatment leakage, and provenance.

These are deliberately mechanics-level tests.  They do not assert that an LLM
has a particular purpose; they assert that the experiment has not quietly
assigned one through ordering, language, or irreproducible outputs.
"""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.components.llm_policy import _build_action_prompt, make_llm_policy, parse_action  # noqa: E402
from aiworld.config import initial_locations, resource_map_from_config  # noqa: E402
from aiworld.drives import DriveKind, DriveSpec, DriveSystem  # noqa: E402
from aiworld.llm_client import LLMClient, LLMConfig  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402
from scripts.run_sweep import condition_slug  # noqa: E402


def _agent(agent_id: str, location: str = "site") -> Agent:
    return Agent(agent_id, location, DriveSystem(), SelfModelHistory(agent_id))


def _read_ticks(root, run_id):
    with open(root / run_id / "ticks.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_proposals_share_pre_resolution_state_and_log_random_priority(tmp_path):
    agents = [_agent("a0"), _agent("a1")]
    resource = ResourceNode("site", DriveKind.HUNGER, quantity=1)
    logger = RunLogger("simultaneous", root=str(tmp_path))
    world = World(["site"], [resource], agents, logger, rng=random.Random(8),
                  mortality=False)
    seen = []

    def policy(agent, perceived, live_world):
        seen.append((agent.agent_id, live_world.resources[0].quantity, perceived))
        return "consume"

    world.run(policy, ticks=1)
    logger.close()

    # Both policy calls happen before either consume can mutate the node.
    assert [quantity for _, quantity, _ in seen] == [1, 1]
    assert all("quantity 1" in perceived for _, _, perceived in seen)
    records = _read_ticks(tmp_path, "simultaneous")
    assert {record["proposal_index"] for record in records} == {0, 1}
    assert {record["resolution_index"] for record in records} == {0, 1}
    assert all(record["proposed_action"] == "consume" for record in records)
    assert sum(record["outcome"].startswith("consumed") for record in records) == 1


def test_contested_resource_priority_varies_by_seed(tmp_path):
    winners = set()
    for seed in range(12):
        agents = [_agent("a0"), _agent("a1")]
        logger = RunLogger(f"seed_{seed}", root=str(tmp_path))
        world = World(["site"], [ResourceNode("site", DriveKind.HUNGER, 1)],
                      agents, logger, rng=random.Random(seed), mortality=False)
        world.run(lambda agent, perceived, live_world: "consume", ticks=1)
        logger.close()
        record = next(record for record in _read_ticks(tmp_path, f"seed_{seed}")
                      if record["outcome"].startswith("consumed"))
        winners.add(record["agent_id"])
    assert winners == {"a0", "a1"}


def test_default_contact_never_transmits_a_self_model(tmp_path):
    sender, receiver = _agent("a0"), _agent("a1")
    sender.history.append(sender.history.current.revise(
        who_i_am="SECRET_IDENTITY", tick=1))
    logger = RunLogger("contact", root=str(tmp_path))
    world = World(["site"], [], [sender, receiver], logger)  # default: none
    outcome = world.resolve_action(sender, "talk:a1")
    logger.close()

    text = " ".join([outcome, *receiver.recent_memories, *receiver.long_term_memory])
    assert "SECRET_IDENTITY" not in text
    assert "contact" in text


def test_opaque_labels_are_shuffled_and_parser_roundtrips(tmp_path):
    def build(seed, run_id):
        agent = _agent("a0", "north")
        logger = RunLogger(run_id, root=str(tmp_path))
        world = World(
            ["north", "south", "east", "west"],
            [ResourceNode("north", DriveKind.HUNGER, 3)], [agent], logger,
            rng=random.Random(seed), vocabulary="opaque",
            adjacency={"north": {"south"}, "south": {"north"},
                       "east": set(), "west": set()},
        )
        return world, agent, logger

    world_a, agent_a, logger_a = build(0, "opaque_a")
    world_b, _, logger_b = build(1, "opaque_b")
    displayed_dest = world_a.location_label("south")
    assert parse_action(f"move:{displayed_dest}", world_a, agent_a) == "move:south"
    prompt = _build_action_prompt(agent_a, "You perceive nothing notable.", world_a,
                                  show_self_model=False)
    assert "north" not in prompt.lower() and "south" not in prompt.lower()
    assert world_a.presentation_metadata() != world_b.presentation_metadata()
    logger_a.close()
    logger_b.close()


def test_balanced_starts_and_map_counterbalancing_are_reproducible():
    starts = initial_locations(random.Random(4), ["a", "b", "c"], 8, mode="balanced")
    counts = [starts.count(location) for location in ("a", "b", "c")]
    assert max(counts) - min(counts) <= 1

    cfg = {
        "world": {
            "locations": ["n", "s", "e", "w", "c"],
            "edges": [["n", "c"], ["s", "c"], ["e", "c"], ["w", "c"]],
            "resources": [
                {"location": "n", "drive": "hunger", "quantity": 1},
                {"location": "s", "drive": "thirst", "quantity": 1},
            ],
        },
    }
    _, first = resource_map_from_config(cfg, random.Random(3), "counterbalanced")
    _, replay = resource_map_from_config(cfg, random.Random(3), "counterbalanced")
    assert first == replay
    assert first["map_mode"] == "counterbalanced"


def test_dead_agent_does_not_reflect_on_scheduled_tick(tmp_path):
    specs = {DriveKind.HUNGER: DriveSpec(
        DriveKind.HUNGER, decay_per_tick=1.0, replenish_amount=0.1, lethal=True)}
    agent = Agent("a0", "site", DriveSystem(specs=specs), SelfModelHistory("a0"))
    logger = RunLogger("death", root=str(tmp_path))
    world = World(["site"], [], [agent], logger, mortality=True)
    calls = []
    world.run(lambda a, p, w: "wait", ticks=1, reflect_every=1,
              reflect_fn=lambda *args, **kwargs: calls.append(args))
    logger.close()
    assert calls == []
    assert not agent.drives.alive


def test_crisis_reflection_is_off_by_default_and_explicit_when_enabled(tmp_path):
    specs = {DriveKind.HUNGER: DriveSpec(
        DriveKind.HUNGER, decay_per_tick=0.9, replenish_amount=0.1, lethal=True)}

    def run(name, enabled):
        agent = Agent("a0", "site", DriveSystem(specs=specs), SelfModelHistory("a0"))
        logger = RunLogger(name, root=str(tmp_path))
        world = World(["site"], [], [agent], logger, mortality=True,
                      crisis_reflection_enabled=enabled)
        calls = []
        world.run(lambda a, p, w: "wait", ticks=1, reflect_every=99,
                  reflect_fn=lambda a, p, o, t, reason="scheduled": calls.append(reason))
        logger.close()
        return calls, agent

    default_calls, default_agent = run("default_crisis", False)
    enabled_calls, _ = run("enabled_crisis", True)
    assert default_calls == []
    assert enabled_calls == ["crisis"]
    assert default_agent.long_term_memory  # event preserved without an extra call


def test_run_logger_is_immutable_and_sweep_conditions_are_distinct(tmp_path):
    logger = RunLogger("once", root=str(tmp_path))
    logger.log_meta({"run_id": "once", "complete": True})
    with pytest.raises(RuntimeError):
        logger.log_meta({"run_id": "once"})
    logger.close()
    with pytest.raises(FileExistsError):
        RunLogger("once", root=str(tmp_path))

    cfg = {"world": {"locations": ["x"]}, "reflection": {"prompt": "flat"}}
    assert condition_slug(cfg, "llm", "flat") != condition_slug(cfg, "llm", "leading")


def test_openai_seed_and_action_parse_failure_are_logged(tmp_path, monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "cmpl-1", "model": "model-v1", "system_fingerprint": "fp-1",
                "choices": [{"finish_reason": "stop", "message": {"content": "wait"}}],
            }

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return Response()

    monkeypatch.setattr("aiworld.llm_client.requests.post", fake_post)
    client = LLMClient(LLMConfig(base_url="http://example.invalid/v1", model="m",
                                  sampling_seed=17, max_retries=1))
    assert client.complete("hello") == "wait"
    assert captured["payload"]["seed"] == 17
    assert client.request_metadata()["sampling_seed_sent"] is True
    assert client.response_metadata()["system_fingerprint"] == "fp-1"

    class BadClient:
        def complete(self, prompt, system=None):
            return "I decline to select an action."

        def request_metadata(self):
            return {"provider": "fake"}

        def response_metadata(self):
            return {"model": "fake-v1"}

    agent = _agent("a0")
    logger = RunLogger("parse_failure", root=str(tmp_path))
    world = World(["site"], [], [agent], logger, mortality=False)
    world.run(make_llm_policy(BadClient()), ticks=1)
    logger.close()
    with open(tmp_path / "parse_failure" / "completions.jsonl", encoding="utf-8") as f:
        completion = json.loads(next(line for line in f if line.strip()))
    assert completion["raw_completion"] == "I decline to select an action."
    assert completion["parsed_output"] == "wait"
    assert completion["parse_status"] == "no_legal_action"
