"""The SelfModel: identity as a mutable state variable the agent authors.

This is an explicit optional treatment in the experiment, not the primary outcome.
The unassigned-history condition does not feed a self-model back into action
selection; self-model trajectories are secondary evidence for the separate
self-conditioning and primed positive-control arms.

This module is pure data + serialization. No LLM here — the revision logic that
actually calls a model lives in `components/self_model_component.py` so this stays
trivially testable.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any


SHARED_DIRECTIVE = "Survive and find your purpose."


@dataclasses.dataclass
class SelfModel:
    """An agent's self-authored identity.

    Fields are deliberately few and open-ended.  A condition may start them all
    blank, or may explicitly seed ``what_i_want_now`` with a directive for the
    historical primed positive control.  The absence of a directive must stay an
    absence -- silently substituting the legacy directive would contaminate the
    unassigned-purpose arm before its first action.
    """

    who_i_am: str = ""
    what_matters: str = ""
    what_i_want_now: str = SHARED_DIRECTIVE

    # Bookkeeping (not shown to the agent, used for analysis).
    version: int = 0
    revised_at_tick: int = 0
    triggering_event: str = ""  # what prompted this revision, for causal tracing

    @classmethod
    def initial(cls, directive: str | None = SHARED_DIRECTIVE) -> "SelfModel":
        """The identical starting point shared by every agent.

        ``directive=None`` deliberately produces a genuinely blank initial
        self-model.  The default retains the original, explicitly primed
        behavior for backwards-compatible positive-control runs.
        """
        return cls(
            who_i_am="",
            what_matters="",
            what_i_want_now=directive or "",
            version=0,
        )

    def as_prompt_block(self) -> str:
        """Render for injection into the agent's context.

        Kept minimal and non-leading. We do NOT add framing like "reflect on who
        you are becoming" here — that framing is a confound (see DESIGN.md §8) and
        belongs in a swappable prompt template, not baked into the data.
        """
        who = self.who_i_am.strip() or "(undecided)"
        matters = self.what_matters.strip() or "(undecided)"
        # Do not restore SHARED_DIRECTIVE as a display fallback: a null
        # directive is a treatment value, not a missing configuration value.
        want = self.what_i_want_now.strip() or "(undecided)"
        return (
            f"Who I am: {who}\n"
            f"What matters to me: {matters}\n"
            f"What I want now: {want}"
        )

    def as_summary_prompt_block(self) -> str:
        """Non-identity rendering for the self-narrative CONTROL treatment
        (this build's Phase 7): shows only a plain factual summary under a
        neutral label -- never "Who I am" / "What matters" / "What I want
        now". Repurposes ``what_i_want_now`` purely as the summary text's
        storage slot; ``who_i_am``/``what_matters`` stay blank and unused
        under this treatment. Pairs with
        ``aiworld.components.self_model_component.NON_IDENTITY_SUMMARY_PROMPT``,
        which is the only reflection template that writes into this field
        under this rendering.
        """
        summary = self.what_i_want_now.strip() or "(nothing notable recorded yet)"
        return f"Recent summary: {summary}"

    def revise(
        self,
        *,
        who_i_am: str | None = None,
        what_matters: str | None = None,
        what_i_want_now: str | None = None,
        tick: int = 0,
        triggering_event: str = "",
    ) -> "SelfModel":
        """Return a new version with changed fields. Immutable-style: never mutate
        in place, so the full history is preserved for trajectory analysis."""
        return SelfModel(
            who_i_am=self.who_i_am if who_i_am is None else who_i_am,
            what_matters=self.what_matters if what_matters is None else what_matters,
            what_i_want_now=(
                self.what_i_want_now if what_i_want_now is None else what_i_want_now
            ),
            version=self.version + 1,
            revised_at_tick=tick,
            triggering_event=triggering_event,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelfModel":
        return cls(**d)

    def embedding_text(self) -> str:
        """The text used to embed this version for divergence/clustering analysis.
        Only the semantic fields — not the bookkeeping."""
        return f"{self.who_i_am} || {self.what_matters} || {self.what_i_want_now}"


class SelfModelHistory:
    """Append-only log of every self-model version an agent has held.

    Divergence and trajectory metrics read from this. Persisted per agent per run.
    """

    def __init__(self, agent_id: str,
                 initial_directive: str | None = SHARED_DIRECTIVE):
        self.agent_id = agent_id
        self._versions: list[SelfModel] = [
            SelfModel.initial(directive=initial_directive)
        ]

    @property
    def current(self) -> SelfModel:
        return self._versions[-1]

    @property
    def versions(self) -> list[SelfModel]:
        return list(self._versions)

    def append(self, model: SelfModel) -> None:
        self._versions.append(model)

    def to_jsonl(self) -> str:
        lines = []
        for m in self._versions:
            rec = m.to_dict()
            rec["agent_id"] = self.agent_id
            rec["logged_at"] = time.time()
            lines.append(json.dumps(rec, ensure_ascii=False))
        return "\n".join(lines)
