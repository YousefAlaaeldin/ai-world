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
    "Recent things you have experienced:\n{recent_memories}\n\n"
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


def parse_revision(text: str, previous: SelfModel, tick: int,
                   triggering_event: str) -> SelfModel:
    """Parse the model's three-line response into a new SelfModel version.

    Robust to extra whitespace / preamble. Falls back to the previous field value if
    a line is missing, so a malformed generation never wipes an identity.
    """
    who = previous.who_i_am
    matters = previous.what_matters
    want = previous.what_i_want_now
    for line in text.splitlines():
        low = line.lower().strip()
        if low.startswith("who i am:"):
            who = line.split(":", 1)[1].strip()
        elif low.startswith("what matters"):
            matters = line.split(":", 1)[1].strip()
        elif low.startswith("what i want"):
            want = line.split(":", 1)[1].strip()
    return previous.revise(
        who_i_am=who, what_matters=matters, what_i_want_now=want,
        tick=tick, triggering_event=triggering_event)


def make_reflect_fn(
    model_call: Callable[[str], str],
    history_by_agent: dict[str, SelfModelHistory],
    logger,
    prompt_template: str = FLAT_REVISION_PROMPT,
    recent_memories_fn: Callable | None = None,
):
    """Build a reflect_fn compatible with World.run(...).

    `model_call(prompt) -> str` is your LLM call (a raw API request, e.g.
    llm_client.LLMClient.complete). We inject it rather than importing a specific
    backend so the same code runs against any model family — which is exactly
    what the cross-model confound check needs.
    """

    def reflect(agent, perceived: str, outcome: str, tick: int,
               reason: str = "scheduled") -> None:
        hist = history_by_agent[agent.agent_id]
        recent = ""
        if recent_memories_fn is not None:
            recent = recent_memories_fn(agent)
        else:
            recent = outcome  # minimal: last outcome as the memory
        prompt = prompt_template.format(
            self_model=hist.current.as_prompt_block(),
            recent_memories=recent,
            bodily_state=agent.drives.as_prompt_block(),
            perceived=perceived,
        )
        response = model_call(prompt)
        # `reason` ("scheduled" | "crisis", see World.run) rides along in the
        # causal-trace field so an event-triggered reflection is distinguishable
        # from a routine periodic one when reading self_models.jsonl later.
        new_model = parse_revision(
            response, hist.current, tick,
            triggering_event=f"[{reason}] {outcome}")
        hist.append(new_model)
        # Point the agent's live history at the updated one.
        agent.history = hist
        logger.log_self_model(agent.agent_id, new_model.to_dict())

    return reflect
