"""The SelfModel: identity as a mutable state variable the agent authors.

This is the conceptual heart of the experiment. Every agent starts with an
(essentially empty) self-model and rewrites it from experience. The *trajectory*
of these documents across time, and their *divergence* across a population that
began identical, is the primary data.

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

    Fields are deliberately few and open-ended. They start nearly empty; only the
    shared directive is given. Everything else the agent writes for itself.
    """

    who_i_am: str = ""
    what_matters: str = ""
    what_i_want_now: str = SHARED_DIRECTIVE

    # Bookkeeping (not shown to the agent, used for analysis).
    version: int = 0
    revised_at_tick: int = 0
    triggering_event: str = ""  # what prompted this revision, for causal tracing

    @classmethod
    def initial(cls) -> "SelfModel":
        """The identical starting point shared by every agent."""
        return cls(
            who_i_am="",
            what_matters="",
            what_i_want_now=SHARED_DIRECTIVE,
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
        want = self.what_i_want_now.strip() or SHARED_DIRECTIVE
        return (
            f"Who I am: {who}\n"
            f"What matters to me: {matters}\n"
            f"What I want now: {want}"
        )

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

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._versions: list[SelfModel] = [SelfModel.initial()]

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
