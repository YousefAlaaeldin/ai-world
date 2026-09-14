"""Regression tests for the unassigned-history condition matrix.

These tests guard experiment semantics, not a model's behavior: a null directive
must remain blank, factual memory must not become identity narration, and the
config files must state every treatment knob explicitly.
"""

from pathlib import Path
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.components.llm_policy import (  # noqa: E402
    ACTION_MEMORY_FACTUAL_LEDGER,
    _action_system_prompt,
    _build_action_prompt,
    _factual_ledger,
    make_llm_policy,
)
from aiworld.config import load_config  # noqa: E402
from aiworld.drives import DriveSystem  # noqa: E402
from aiworld.logging_schema import RunLogger  # noqa: E402
from aiworld.self_model import SHARED_DIRECTIVE, SelfModelHistory  # noqa: E402
from aiworld.sim.world import Agent, World  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _world(tmp_path, *, history=None, memory_style="factual"):
    agent = Agent(
        "agent_0", "site", DriveSystem(),
        history or SelfModelHistory("agent_0"),
    )
    logger = RunLogger("condition", root=str(tmp_path))
    return World(["site"], [], [agent], logger, mortality=False,
                 memory_style=memory_style), agent, logger


def test_null_directive_initializes_a_genuinely_blank_self_model():
    history = SelfModelHistory("agent_0", initial_directive=None)

    assert history.current.who_i_am == ""
    assert history.current.what_matters == ""
    assert history.current.what_i_want_now == ""
    prompt_block = history.current.as_prompt_block().lower()
    assert "survive" not in prompt_block
    assert "purpose" not in prompt_block
    assert prompt_block.count("(undecided)") == 3


def test_explicit_positive_control_still_seeds_legacy_directive():
    history = SelfModelHistory("agent_0", initial_directive=SHARED_DIRECTIVE)
    assert history.current.what_i_want_now == SHARED_DIRECTIVE
    assert SHARED_DIRECTIVE in history.current.as_prompt_block()


def test_unassigned_self_narrative_starts_blank_not_with_legacy_directive(tmp_path):
    history = SelfModelHistory("agent_0", initial_directive=None)
    world, agent, logger = _world(tmp_path, history=history)
    system = _action_system_prompt(
        "neutral", directive=None, show_self_model=True)
    prompt = _build_action_prompt(
        agent, "You perceive nothing notable.", world,
        show_self_model=True,
    )
    logger.close()

    text = system + "\n" + prompt
    assert "Who I am: (undecided)" in text
    assert "What I want now: (undecided)" in text
    assert "survive" not in text.lower()
    assert "purpose" not in text.lower()


def test_factual_ledger_is_bounded_and_contains_no_identity_block(tmp_path):
    history = SelfModelHistory("agent_0", initial_directive=None)
    history.append(history.current.revise(
        who_i_am="an injected identity", what_matters="a narrative goal",
        what_i_want_now="a named purpose", tick=1,
    ))
    world, agent, logger = _world(tmp_path, history=history)
    agent.long_term_memory.extend([
        "tick 1: R1 depleted at L1",
        "tick 2: received 1 R2 from agent_1",
    ])
    agent.recent_memories.extend([
        "tick 2: did 'wait' -> waited",
        "tick 3: did 'move:site' -> tried to move to unreachable place site",
    ])

    prompt = _build_action_prompt(
        agent, "You perceive nothing notable.", world,
        show_self_model=False,
        action_memory=ACTION_MEMORY_FACTUAL_LEDGER,
        action_memory_max_events=3,
        action_context_max_chars=200,
    )
    logger.close()

    assert "Factual event ledger (oldest to newest):" in prompt
    assert "tick 1:" not in prompt  # newest three entries only
    assert "tick 2:" in prompt and "tick 3:" in prompt
    for forbidden in ("Who I am:", "What matters", "What I want", "identity",
                      "purpose", "remember", "injected identity"):
        assert forbidden.lower() not in prompt.lower()


def test_factual_ledger_hard_caps_single_oversized_event(tmp_path):
    world, agent, logger = _world(tmp_path)
    agent.recent_memories.append("tick 1: " + "x" * 500)
    ledger = _factual_ledger(agent, max_events=12, max_chars=96)
    logger.close()

    assert len(ledger) <= 96
    assert ledger.startswith("Factual event ledger")
    assert ledger.endswith("…")


def test_factual_memory_reaches_the_next_action_without_self_model(tmp_path):
    class RecordingClient:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt, system=None):
            self.prompts.append((system, prompt))
            return "wait"

    world, _, logger = _world(tmp_path)
    client = RecordingClient()
    policy = make_llm_policy(
        client,
        directive=None,
        show_self_model=False,
        action_memory=ACTION_MEMORY_FACTUAL_LEDGER,
        action_memory_max_events=12,
        action_context_max_chars=1_200,
    )
    world.run(policy, ticks=2)
    logger.close()

    first_system, first_prompt = client.prompts[0]
    second_system, second_prompt = client.prompts[1]
    assert "Factual event ledger" in first_prompt
    assert "(no recorded events)" in first_prompt
    assert "tick 1: did 'wait' -> waited" in second_prompt
    for text in (first_system + first_prompt, second_system + second_prompt):
        for forbidden in ("survive", "purpose", "self-description", "Who I am"):
            assert forbidden.lower() not in text.lower()


def test_unassigned_condition_configs_have_explicit_distinct_treatments():
    factual = load_config(PROJECT_ROOT / "configs" /
                          "experiment_factual_memory_unassigned.yaml")
    narrative = load_config(PROJECT_ROOT / "configs" /
                            "experiment_self_narrative_unassigned.yaml")
    baseline = load_config(PROJECT_ROOT / "configs" /
                           "experiment_deprimed_baseline.yaml")
    ecological = load_config(PROJECT_ROOT / "configs" /
                             "experiment_null_baseline.yaml")

    for cfg in (factual, narrative, baseline, ecological):
        agents = cfg["agents"]
        assert "show_self_model" in agents
        assert "directive" in agents
        assert "action_memory" in agents
        assert "action_memory_max_events" in agents
        assert "action_context_max_chars" in agents

    assert factual["agents"]["directive"] is None
    assert factual["agents"]["show_self_model"] is False
    assert factual["agents"]["action_memory"] == "factual_ledger"
    assert factual["world"]["memory_style"] == "factual"
    assert factual["reflection"]["enabled"] is False
    assert len(factual["run"]["seeds"]) == 20

    assert narrative["agents"]["directive"] is None
    assert narrative["agents"]["show_self_model"] is True
    assert narrative["agents"]["action_memory"] == "none"
    assert narrative["world"]["memory_style"] == "factual"
    assert narrative["reflection"]["enabled"] is True
    assert len(narrative["run"]["seeds"]) == 20

    assert baseline["agents"]["action_memory"] == "none"
    assert ecological["agents"]["directive"] is None
    assert ecological["agents"]["show_self_model"] is False
    assert ecological["reflection"]["enabled"] is False


def test_runner_records_resolved_factual_condition_and_blank_initial_model(
        tmp_path, monkeypatch):
    """The provenance record must describe what was actually passed to the LLM."""
    import scripts.run_llm as run_llm

    class TempLogger(RunLogger):
        def __init__(self, run_id):
            super().__init__(run_id, root=str(tmp_path))

    class FakeConfig:
        model = "fake-model"
        base_url = "http://fake.invalid/v1"
        provider = "fake"
        sampling_seed = None

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def complete(self, prompt, system=None):
            return "wait"

        def request_metadata(self):
            return {"model": self.config.model, "sampling_seed": self.config.sampling_seed}

        def response_metadata(self):
            return {}

        def stats(self):
            return {"n_calls": 1}

    monkeypatch.setattr(run_llm, "RunLogger", TempLogger)
    monkeypatch.setattr(run_llm, "LLMClient", FakeClient)
    monkeypatch.setattr(run_llm, "config_from_env", FakeConfig)
    monkeypatch.setattr(sys, "argv", [
        "run_llm.py",
        "--config", str(PROJECT_ROOT / "configs" /
                         "experiment_factual_memory_unassigned.yaml"),
        "--agents", "1",
        "--ticks", "2",
        "--run-id", "metadata_check",
    ])

    run_llm.main()

    meta = json.loads((tmp_path / "metadata_check" / "meta.json").read_text())
    condition = meta["condition"]
    assert condition["directive"] is None
    assert condition["initial_self_model_is_blank"] is True
    assert condition["action_memory"] == "factual_ledger"
    assert condition["action_memory_max_events"] == 12
    assert condition["action_context_max_chars"] == 1200
    assert condition["world_memory_style"] == "factual"
    assert set(meta["randomization_seeds"]) == {
        "resource_map", "initial_locations", "presentation_and_resolution",
    }
    assert meta["sampling_seed"] == 0
    assert meta["resolved_config"]["agents"]["directive"] is None
    assert (tmp_path / "metadata_check" / "self_models.jsonl").read_text().splitlines()[0]
    initial = json.loads(
        (tmp_path / "metadata_check" / "self_models.jsonl").read_text().splitlines()[0])
    assert initial["what_i_want_now"] == ""
