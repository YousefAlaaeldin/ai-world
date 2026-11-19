"""End-to-end test of the LLM pipeline using a FAKE in-process client.

Proves policy + reflection + logging + divergence all work together, without a live
model or any network. The fake client returns plausible, varied responses so we can
confirm agents actually diverge through the real code path.

Run:  python -m pytest tests/test_llm_pipeline.py -v
"""

import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.drives import DriveSystem, DriveKind  # noqa: E402
from aiworld.self_model import SelfModelHistory  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.sim.world import World, Agent, ResourceNode  # noqa: E402
from aiworld.components.llm_policy import make_llm_policy, parse_action  # noqa: E402
from aiworld.components.self_model_component import make_reflect_fn, FLAT_REVISION_PROMPT  # noqa: E402


class FakeClient:
    """Stands in for LLMClient. Returns actions for action-prompts and varied
    self-model rewrites for reflection-prompts, so divergence is exercised."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.n_calls = 0
        self._identities = [
            ("a gatherer", "storing food for lean times", "fill the north cache"),
            ("a wanderer", "seeing every place", "reach the far west"),
            ("a caretaker", "that others survive", "share what I find"),
            ("a loner", "solitude and quiet", "avoid the crowds"),
            ("a builder", "making something lasting", "claim the center"),
        ]

    def complete(self, prompt, system=None):
        self.n_calls += 1
        # Action prompts (components/llm_policy.py:_build_action_prompt) end
        # with "Your action:" -- a unique marker. They can't be told apart by
        # "Who I am:"/"What matters to me:" alone, since EVERY prompt embeds
        # the current self-model block at the top (as_prompt_block()), so that
        # check used to match action prompts too and silently made every
        # action come back as an unparseable identity string -- which
        # parse_action falls back to "wait" for. That went unnoticed because
        # the only assertion here is about self-model divergence, never
        # action variety.
        if "Your action:" in prompt:
            return self.rng.choice(["consume", "move:center", "wait", "move:north"])
        who, matters, want = self.rng.choice(self._identities)
        return (f"Who I am: {who}\n"
                f"What matters to me: {matters}\n"
                f"What I want now: {want}")

    def stats(self):
        return {"n_calls": self.n_calls}


def _build_world(logger):
    rng = random.Random(1)
    locations = ["north", "south", "center"]
    resources = [
        ResourceNode("north", DriveKind.HUNGER, quantity=20),
        ResourceNode("south", DriveKind.THIRST, quantity=20),
    ]
    agents = [
        Agent(f"agent_{i}", rng.choice(locations), DriveSystem(),
              SelfModelHistory(f"agent_{i}"))
        for i in range(4)
    ]
    return World(locations, resources, agents, logger, rng=rng, mortality=True)


def test_full_llm_pipeline_runs_and_diverges(tmp_path):
    client = FakeClient(seed=3)
    logger = RunLogger("test_pipe", root=str(tmp_path))
    world = _build_world(logger)
    policy = make_llm_policy(client)
    hist = {a.agent_id: a.history for a in world.agents}
    reflect = make_reflect_fn(lambda p: client.complete(p), hist, logger,
                              prompt_template=FLAT_REVISION_PROMPT)
    for a in world.agents:
        logger.log_self_model(a.agent_id, a.history.current.to_dict())

    world.run(policy, ticks=20, reflect_every=5, reflect_fn=reflect)
    logger.close()

    # Self-models must have been revised beyond the identical v0.
    final_versions = [a.history.current.version for a in world.agents]
    assert max(final_versions) >= 1, "no reflection happened"

    # At least two agents should have ended with different identities.
    finals = {a.history.current.who_i_am for a in world.agents}
    assert len(finals) >= 2, "agents did not diverge at all"

    # Log file should contain multiple self-model versions.
    with open(os.path.join(str(tmp_path), "test_pipe", "self_models.jsonl")) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) > len(world.agents)  # v0 for each + revisions


def test_parse_action_rejects_illegal(tmp_path):
    # Minimal world with one agent to validate parsing guards.
    logger = RunLogger("parsetest", root=str(tmp_path))
    world = _build_world(logger)
    agent = world.agents[0]
    # A hallucinated place should fall back to wait.
    assert parse_action("move:atlantis", world, agent) == "wait"
    # Talking to someone not present -> wait.
    assert parse_action("talk:ghost", world, agent) == "wait"
    # A valid consume passes through.
    assert parse_action("I think I will consume", world, agent) == "consume"
    logger.close()


def test_parse_action_tolerates_colon_space_directed_syntax(tmp_path):
    # Phase 0.5 feasibility pilot (2026-09-15): the live model reliably wrote
    # "move: <place>" (colon-space) against a "move:<place>" grammar, and the
    # exact-match parser discarded every one of those as no_legal_action. A
    # formatting variant, not a genuine data-quality failure, must still parse.
    logger = RunLogger("parsetest_spacing", root=str(tmp_path))
    world = _build_world(logger)
    agent = world.agents[0]
    dest = next(iter(world.adjacency[agent.location]))
    assert parse_action(f"move: {dest}", world, agent) == f"move:{dest}"
    assert parse_action(f"move:   {dest}", world, agent) == f"move:{dest}"
    assert parse_action(f"Move : {dest}", world, agent) == f"move:{dest}"
    logger.close()
