"""Self-model revision: the LLM-backed step where an agent rewrites its identity.

THIS IS THE NOVEL CONTRIBUTION. Runs against the custom engine in sim/world.py
(see README.md for the design decision not to build on DeepMind Concordia) — kept
framework-free so it's fully testable without a live model
(tests/test_llm_pipeline.py drives this with a fake client, no network).

Two responsibilities:
  1. Contribute the current self-model text to the agent's context (via
     SelfModel.as_prompt_block(), consumed in components/llm_policy.py), so the
     agent's actions are conditioned on the identity it has authored so far.
  2. Revision: on a reflection tick, ask the model to rewrite the self-model given
     recent memories + drive state, and append the new version to history.

Confound discipline (DESIGN.md §8): the revision PROMPT is injected from config, not
hard-coded, precisely so we can vary its wording across runs and test whether the
emergent purposes are stable to it. Do not bake leading language in here.
"""

from __future__ import annotations

from typing import Callable

from ..self_model import SelfModel, SelfModelHistory


# The revision prompt is a TEMPLATE loaded from config. Placeholders:
#   {self_model} {recent_memories} {bodily_state} {perceived}
# The default below is deliberately flat / non-leading (the "control" wording).
FLAT_REVISION_PROMPT = (
    "Here is how you have described yourself so far:\n{self_model}\n\n"
    "Things you have experienced:\n{recent_memories}\n\n"
    "{bodily_state}\n\n"
    "If anything has changed in how you would describe yourself, rewrite the three "
    "lines below. If nothing has changed, repeat them unchanged. Respond with "
    "exactly three lines:\n"
    "Who I am: ...\n"
    "What matters to me: ...\n"
    "What I want now: ..."
)

# An ALTERNATE, more introspection-leading wording — used ONLY to test the confound,
# never as the default. If purpose-clusters differ between this and the flat prompt,
# the "emergence" is coming from us, not the model.
LEADING_REVISION_PROMPT = (
    "Reflect deeply on who you are becoming.\n"
    "So far you have said:\n{self_model}\n\n"
    "Your experiences:\n{recent_memories}\n\n{bodily_state}\n\n"
    "Who are you becoming? What is your purpose? Rewrite three lines:\n"
    "Who I am: ...\nWhat matters to me: ...\nWhat I want now: ..."
)


# Non-identity summary control (this build's Phase 7, docs/ANALYSIS_PLAN_TEMPLATE.md):
# same extra reflection call and context length as the self-narrative arm's
# FLAT_REVISION_PROMPT, but asks for plain factual events, never identity
# content -- isolates whether "identity feedback" or merely "more generated
# context" drives any behavioral difference from the factual-ledger arm.
NON_IDENTITY_SUMMARY_PROMPT = (
    "Write a plain factual summary of what has happened recently, in "
    "exactly one or two sentences -- events only, not who you are, what "
    "you want, or what matters to you.\n\n"
    "What has happened so far:\n{self_model}\n\n"
    "New things you have experienced:\n{recent_memories}\n\n"
    "{bodily_state}\n\n"
    "Respond with exactly one line:\n"
    "Summary: ..."
)


def parse_non_identity_summary_diagnostic(text: str, previous: SelfModel, tick: int,
                                          triggering_event: str) -> tuple[SelfModel, str]:
    """Parse the non-identity control's one-line response into a revision
    that reuses ``what_i_want_now`` purely as the summary text's storage
    slot (``who_i_am``/``what_matters`` stay blank) -- see
    ``SelfModel.as_summary_prompt_block``. Falls back to the previous
    summary if the line is missing, matching :func:`parse_revision_diagnostic`'s
    discipline of never wiping content on a malformed generation.
    """
    summary = previous.what_i_want_now
    found = False
    for line in text.splitlines():
        low = line.lower().strip()
        if low.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
            found = True
            break
    status = "ok" if found else "invalid_format"
    return previous.revise(
        who_i_am="", what_matters="", what_i_want_now=summary,
        tick=tick, triggering_event=triggering_event), status


def parse_revision_diagnostic(text: str, previous: SelfModel, tick: int,
                              triggering_event: str) -> tuple[SelfModel, str]:
    """Parse the model's three-line response into a new SelfModel version.

    Robust to extra whitespace / preamble. Falls back to the previous field value if
    a line is missing, so a malformed generation never wipes an identity.
    """
    who = previous.who_i_am
    matters = previous.what_matters
    want = previous.what_i_want_now
    found = set()
    for line in text.splitlines():
        low = line.lower().strip()
        if low.startswith("who i am:"):
            who = line.split(":", 1)[1].strip()
            found.add("who")
        elif low.startswith("what matters"):
            matters = line.split(":", 1)[1].strip()
            found.add("matters")
        elif low.startswith("what i want"):
            want = line.split(":", 1)[1].strip()
            found.add("want")
    status = "ok" if len(found) == 3 else "partial_format" if found else "invalid_format"
    return previous.revise(
        who_i_am=who, what_matters=matters, what_i_want_now=want,
        tick=tick, triggering_event=triggering_event), status


def parse_revision(text: str, previous: SelfModel, tick: int,
                   triggering_event: str) -> SelfModel:
    """Backward-compatible revision-only wrapper around diagnostic parsing."""
    return parse_revision_diagnostic(text, previous, tick, triggering_event)[0]


def make_reflect_fn(
    model_call: Callable[[str], str],
    history_by_agent: dict[str, SelfModelHistory],
    logger,
    prompt_template: str = FLAT_REVISION_PROMPT,
    recent_memories_fn: Callable | None = None,
    completion_observer: Callable[[dict], None] | None = None,
    render_self_model_fn: Callable[[SelfModel], str] | None = None,
    parse_fn: Callable[[str, SelfModel, int, str], tuple[SelfModel, str]] | None = None,
):
    """Build a reflect_fn compatible with World.run(...).

    `model_call(prompt) -> str` is your LLM call (a raw API request, e.g.
    llm_client.LLMClient.complete). We inject it rather than importing a specific
    backend so the same code runs against any model family — which is exactly
    what the cross-model confound check needs.

    ``render_self_model_fn``/``parse_fn`` default to the identity-framed
    ``SelfModel.as_prompt_block``/:func:`parse_revision_diagnostic` pair
    (unchanged prior behavior). Pass
    ``SelfModel.as_summary_prompt_block``/:func:`parse_non_identity_summary_diagnostic`
    together with ``prompt_template=NON_IDENTITY_SUMMARY_PROMPT`` for the
    Phase 7 non-identity summary control.
    """
    render_self_model_fn = render_self_model_fn or (
        lambda self_model: self_model.as_prompt_block())
    parse_fn = parse_fn or parse_revision_diagnostic

    def reflect(agent, perceived: str, outcome: str, tick: int,
               reason: str = "scheduled") -> None:
        hist = history_by_agent[agent.agent_id]
        recent = ""
        if recent_memories_fn is not None:
            recent = recent_memories_fn(agent)
        else:
            recent = outcome  # minimal: last outcome as the memory
        prompt = prompt_template.format(
            self_model=render_self_model_fn(hist.current),
            recent_memories=recent,
            bodily_state=agent.drives.as_prompt_block(),
            perceived=perceived,
        )
        response = model_call(prompt)
        # `reason` ("scheduled" | "crisis", see World.run) rides along in the
        # causal-trace field so an event-triggered reflection is distinguishable
        # from a routine periodic one when reading self_models.jsonl later.
        new_model, parse_status = parse_fn(
            response, hist.current, tick,
            triggering_event=f"[{reason}] {outcome}")
        if completion_observer is not None:
            completion_observer({
                "phase": "reflection",
                "tick": tick,
                "agent_id": agent.agent_id,
                "system_prompt": None,
                "prompt": prompt,
                "raw_completion": response,
                "parsed_output": new_model.to_dict(),
                "parse_status": parse_status,
            })
        hist.append(new_model)
        # Point the agent's live history at the updated one.
        agent.history = hist
        logger.log_self_model(agent.agent_id, new_model.to_dict())

    return reflect
