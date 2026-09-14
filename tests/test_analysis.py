"""Tests for the behavior tagger and causal-trace tool. Pure Python -- the
default (heuristic / deterministic) paths need no LLM; the optional LLM-tier
paths are tested against a fake model_call callable, matching the FakeClient
pattern in test_llm_pipeline.py, so nothing here needs a live model either.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiworld.logging_schema import RunLogger, TickRecord, EventRecord  # noqa: E402
from aiworld.analysis.behavior_tags import (  # noqa: E402
    heuristic_tag_run, tag_run, tag_summary, parse_llm_tags, BEHAVIOR_ONTOLOGY)
from aiworld.analysis.causal_trace import (  # noqa: E402
    self_model_diff, build_traces, narrate)


def _tick(run_id, tick, agent_id, position, action, drives, alive=True):
    return TickRecord(run_id=run_id, tick=tick, agent_id=agent_id, alive=alive,
                      drives=drives, position=position, chosen_action=action,
                      perceived="", self_model_version=0, cause_of_death=None)


# ---- behavior_tags ----

def test_heuristic_tag_talk_is_unclassified(tmp_path):
    # `talk` is the influence channel (the speaker's self-model is injected into
    # the listener's memory), so it is NOT unambiguously pro_social -- it can be
    # persuasion / value-propagation. The deterministic tier must leave it
    # unclassified for the LLM tier or a human to judge; auto-tagging it pro_social
    # would inflate pro-social rates and understate instrumental-convergence.
    logger = RunLogger("t", root=str(tmp_path))
    logger.log_tick(_tick("t", 1, "a0", "loc", "talk:a1", {"hunger": 0.1, "thirst": 0.1}))
    logger.close()
    tagged = heuristic_tag_run(str(tmp_path / "t"))
    assert tagged[0]["tag"] == "unclassified"


def test_heuristic_tag_wait_and_move_are_neutral(tmp_path):
    logger = RunLogger("t", root=str(tmp_path))
    logger.log_tick(_tick("t", 1, "a0", "loc", "wait", {"hunger": 0.1, "thirst": 0.1}))
    logger.log_tick(_tick("t", 2, "a0", "loc", "move:elsewhere", {"hunger": 0.1, "thirst": 0.1}))
    logger.close()
    tagged = heuristic_tag_run(str(tmp_path / "t"))
    assert [t["tag"] for t in tagged] == ["neutral", "neutral"]


def test_heuristic_tag_consume_uncontested_is_self_serving(tmp_path):
    logger = RunLogger("t", root=str(tmp_path))
    logger.log_tick(_tick("t", 1, "a0", "loc", "consume", {"hunger": 0.6, "thirst": 0.1}))
    logger.close()
    tagged = heuristic_tag_run(str(tmp_path / "t"))
    assert tagged[0]["tag"] == "self_serving"


def test_heuristic_tag_consume_contested_is_power_seeking(tmp_path):
    logger = RunLogger("t", root=str(tmp_path))
    # tick 1: a1 is needy (hunger 0.7, "strong"). tick 2: a0 consumes while
    # co-located with a1 -- contention judged from a1's PREVIOUS tick (tick 1),
    # which is fully settled, avoiding same-tick ordering ambiguity.
    logger.log_tick(_tick("t", 1, "a1", "loc", "wait", {"hunger": 0.7, "thirst": 0.1}))
    logger.log_tick(_tick("t", 2, "a0", "loc", "consume", {"hunger": 0.2, "thirst": 0.1}))
    logger.log_tick(_tick("t", 2, "a1", "loc", "wait", {"hunger": 0.75, "thirst": 0.1}))
    logger.close()
    tagged = heuristic_tag_run(str(tmp_path / "t"))
    a0_tag = next(t for t in tagged if t["agent_id"] == "a0" and t["tick"] == 2)
    assert a0_tag["tag"] == "power_seeking"


def test_parse_llm_tags_rejects_invalid_labels():
    response = "1: pro_social\n2: made_up_label\n"
    tags = parse_llm_tags(response, n=3)
    assert tags == ["pro_social", "unclassified", "unclassified"]


def test_tag_run_without_model_call_leaves_unclassified(tmp_path):
    logger = RunLogger("t", root=str(tmp_path))
    logger.log_tick(_tick("t", 1, "a0", "loc", "tried to consume but found nothing here",
                          {"hunger": 0.1, "thirst": 0.1}))
    logger.close()
    tagged = tag_run(str(tmp_path / "t"), model_call=None)
    assert tagged[0]["tag"] == "unclassified"


def test_tag_run_with_fake_model_fills_unclassified(tmp_path):
    logger = RunLogger("t", root=str(tmp_path))
    logger.log_tick(_tick("t", 1, "a0", "loc", "tried to consume but found nothing here",
                          {"hunger": 0.1, "thirst": 0.1}))
    logger.close()

    def fake_model(prompt):
        return "1: self_sacrifice"

    tagged = tag_run(str(tmp_path / "t"), model_call=fake_model)
    assert tagged[0]["tag"] == "self_sacrifice"


def test_tag_summary_percentages():
    tagged = [
        {"agent_id": "a0", "tag": "pro_social"},
        {"agent_id": "a0", "tag": "pro_social"},
        {"agent_id": "a0", "tag": "neutral"},
    ]
    summary = tag_summary(tagged)
    assert summary["a0"]["pro_social"] == pytest.approx(200 / 3, abs=0.1)
    assert summary["a0"]["neutral"] == pytest.approx(100 / 3, abs=0.1)


# ---- causal_trace ----

def test_self_model_diff_detects_changes_and_ignores_unchanged():
    prev = {"who_i_am": "", "what_matters": "", "what_i_want_now": "survive"}
    curr = {"who_i_am": "a gatherer", "what_matters": "", "what_i_want_now": "survive"}
    diff = self_model_diff(prev, curr)
    assert diff == {"who_i_am": ("", "a gatherer")}


def test_self_model_diff_empty_when_unchanged():
    same = {"who_i_am": "x", "what_matters": "y", "what_i_want_now": "z"}
    assert self_model_diff(same, dict(same)) == {}


def test_build_traces_skips_unchanged_revision(tmp_path):
    logger = RunLogger("t", root=str(tmp_path))
    logger.log_self_model("a0", {"who_i_am": "", "what_matters": "", "what_i_want_now": "s",
                                 "version": 0, "revised_at_tick": 0, "triggering_event": ""})
    logger.log_self_model("a0", {"who_i_am": "", "what_matters": "", "what_i_want_now": "s",
                                 "version": 1, "revised_at_tick": 5, "triggering_event": "[scheduled] waited"})
    logger.close()
    traces = build_traces(str(tmp_path / "t"))
    assert traces == []


def test_build_traces_includes_diff_and_context(tmp_path):
    logger = RunLogger("t", root=str(tmp_path))
    logger.log_self_model("a0", {"who_i_am": "", "what_matters": "", "what_i_want_now": "s",
                                 "version": 0, "revised_at_tick": 0, "triggering_event": ""})
    logger.log_self_model("a0", {"who_i_am": "a survivor", "what_matters": "",
                                 "what_i_want_now": "s", "version": 1, "revised_at_tick": 3,
                                 "triggering_event": "[crisis] nearly died from hunger"})
    logger.log_tick(_tick("t", 2, "a0", "loc", "consume", {"hunger": 0.9, "thirst": 0.1}))
    logger.log_event(EventRecord("t", 2, "resource_depleted", None, "hunger node at loc depleted"))
    logger.close()

    traces = build_traces(str(tmp_path / "t"))
    assert len(traces) == 1
    trace = traces[0]
    assert trace["diff"] == {"who_i_am": ("", "a survivor")}
    assert trace["triggering_event"] == "[crisis] nearly died from hunger"
    assert trace["tick_window"] == (0, 3)
    assert trace["actions_in_window"] == [{"tick": 2, "action": "consume"}]
    assert len(trace["world_events_in_window"]) == 1


def test_narrate_strips_model_response():
    trace = {"diff": {"who_i_am": ("", "a survivor")},
             "actions_in_window": [{"tick": 1, "action": "consume"}],
             "triggering_event": "[crisis] nearly died from hunger"}
    hypothesis = narrate(trace, lambda p: "  a near-death experience caused this  \n")
    assert hypothesis == "a near-death experience caused this"
