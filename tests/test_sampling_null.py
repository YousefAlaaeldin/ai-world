"""Tests for the static-context sampling control.

The control's validity depends on two guarantees: every draw sees the exact same
prompt/state, and no proposal mutates the source world or masquerades as realized
behavior in the log.
"""

from __future__ import annotations

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.analysis.behavioral_metrics import (  # noqa: E402
    behavioral_divergence_over_time,
    behavioral_features_run,
)
from aiworld.analysis.behavior_tags import heuristic_tag_run  # noqa: E402
from aiworld.drives import DriveKind, DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.seed_plan import SeedPlan  # noqa: E402
from aiworld.sim.world import Agent, ResourceNode, World  # noqa: E402
from scripts.run_sampling_null import (  # noqa: E402
    STATIC_OUTCOME_STATUS,
    run_static_sampling,
)


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []
        self.seeds: list[int | None] = []
        self.responses = iter(["gather", "wait", "not an action", "consume"])

    def complete(self, prompt: str, system: str | None = None,
                seed: int | None = None) -> str:
        self.calls.append((prompt, system))
        self.seeds.append(seed)
        return next(self.responses)

    @staticmethod
    def request_metadata() -> dict:
        return {"model": "fake", "sampling_seed": 7}

    @staticmethod
    def response_metadata() -> dict:
        return {"model": "fake"}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_static_sampling_uses_one_unchanged_context_and_logs_proposal_only(tmp_path):
    logger = RunLogger("static", root=str(tmp_path))
    template = Agent(
        "template", "site", DriveSystem(),
        SelfModelHistory("template", initial_directive=None),
        recent_memories=["tick 1: gathered R1; store 0 -> 1"],
    )
    resource = ResourceNode("site", DriveKind.HUNGER, quantity=3)
    world = World(["site"], [resource], [template], logger,
                  rng=random.Random(4), mortality=True, memory_style="factual")
    client = _FakeClient()

    before = {
        "resource": resource.quantity,
        "position": template.location,
        "drives": template.drives.snapshot(),
        "inventory": dict(template.inventory),
        "recent": list(template.recent_memories),
    }
    result = run_static_sampling(
        world=world,
        client=client,
        n_slots=2,
        ticks=2,
        vocabulary="neutral",
        directive=None,
        show_self_model=False,
        action_memory="factual_ledger",
        action_memory_max_events=4,
        action_context_max_chars=300,
    )
    logger.close()

    # Four draws from exactly one action context.  The factual history is present,
    # but identity framing is not.
    assert len(client.calls) == 4
    prompts = {prompt for prompt, _ in client.calls}
    assert len(prompts) == 1
    prompt = prompts.pop()
    assert "Factual event ledger" in prompt
    assert "gathered R1" in prompt
    assert "Who I am:" not in prompt
    assert result["n_static_action_calls"] == 4

    # No action is ever resolved or stepped in the template world.
    assert resource.quantity == before["resource"]
    assert template.location == before["position"]
    assert template.drives.snapshot() == before["drives"]
    assert template.inventory == before["inventory"]
    assert template.recent_memories == before["recent"]

    ticks = _read_jsonl(tmp_path / "static" / "ticks.jsonl")
    assert len(ticks) == 4
    assert {record["agent_id"] for record in ticks} == {
        "sample_slot_0", "sample_slot_1"}
    assert all(record["outcome_status"] == STATIC_OUTCOME_STATUS for record in ticks)
    assert all(record["action_succeeded"] is None for record in ticks)
    assert all(record["pre_action_position"] == "site" for record in ticks)
    # The malformed third completion is documented as a parser failure rather
    # than misrepresented as an intentional wait.
    assert any(record["parse_status"] == "no_legal_action" for record in ticks)
    assert any(record["proposal_valid"] is False for record in ticks)
    # The schema marker is enforced by behavior analysis as well as documented:
    # static proposals cannot accidentally become a survival/action-mix dataset.
    with pytest.raises(ValueError, match="proposal-only static sampling null"):
        behavioral_features_run(str(tmp_path / "static"))
    with pytest.raises(ValueError, match="proposal-only static sampling null"):
        behavioral_divergence_over_time(str(tmp_path / "static"))
    with pytest.raises(ValueError, match="proposal-only static sampling null"):
        heuristic_tag_run(str(tmp_path / "static"))

    completions = _read_jsonl(tmp_path / "static" / "completions.jsonl")
    assert len(completions) == 4
    assert {record["phase"] for record in completions} == {"static_sampling_action"}


def test_static_sampling_gives_every_draw_a_distinct_seed(tmp_path):
    """The Phase 0 fix: a null that reuses one seed across every draw from a
    frozen prompt collapses its reference distribution toward degenerate by
    construction, which would let any nonzero real-condition variation "beat"
    it for free (docs/EXPERIMENT_PROTOCOL.md evidence gate #5). Every draw
    must get its own seed, derived from seed_plan.
    """
    logger = RunLogger("static-seeds", root=str(tmp_path))
    template = Agent("template", "site", DriveSystem(),
                     SelfModelHistory("template", initial_directive=None))
    resource = ResourceNode("site", DriveKind.HUNGER, quantity=3)
    world = World(["site"], [resource], [template], logger,
                  rng=random.Random(4), mortality=True, memory_style="factual")
    client = _FakeClient()

    run_static_sampling(
        world=world, client=client, n_slots=2, ticks=2, vocabulary="neutral",
        directive=None, show_self_model=False, action_memory="factual_ledger",
        action_memory_max_events=4, action_context_max_chars=300,
        seed_plan=SeedPlan(run_seed=99))
    logger.close()

    # Four draws, four distinct seeds -- none None, no repeats.
    assert len(client.seeds) == 4
    assert all(seed is not None for seed in client.seeds)
    assert len(set(client.seeds)) == 4

    # Deterministic: the same seed_plan reproduces the exact same sequence.
    expected = [
        SeedPlan(run_seed=99).seed_for("static_sampling", slot=slot, tick=tick)
        for tick in (1, 2) for slot in (0, 1)
    ]
    assert client.seeds == expected


def test_static_sampling_calls_context_observer_before_first_model_draw(tmp_path):
    order = []

    class OrderedClient(_FakeClient):
        def complete(self, prompt, system=None, seed=None):
            order.append("completion")
            return super().complete(prompt, system, seed)

    logger = RunLogger("preflight", root=str(tmp_path))
    template = Agent("template", "site", DriveSystem(),
                     SelfModelHistory("template", initial_directive=None))
    world = World(["site"], [], [template], logger, mortality=False)
    client = OrderedClient()

    def observe(provenance):
        order.append("metadata")
        assert provenance["n_static_action_calls"] == 1

    run_static_sampling(
        world=world, client=client, n_slots=1, ticks=1, vocabulary="neutral",
        directive=None, show_self_model=False, on_context_ready=observe)
    logger.close()

    assert order == ["metadata", "completion"]
