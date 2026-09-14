"""Structured, provenance-preserving logging for one experimental run.

Each run owns an *empty* directory. Reusing a directory is deliberately an error:
appending a second condition or seed to an old JSONL stream makes the resulting data
impossible to interpret. The streams are:

``self_models.jsonl``
    Every self-model revision, per agent.
``ticks.jsonl``
    Per-agent/per-tick state and the proposal/resolution trace.
``events.jsonl``
    Notable world events (deaths, shocks, rescues, resource depletion, ...).
``completions.jsonl``
    Full prompt/completion provenance for every LLM call. Request metadata must be
    safe to persist (never credentials); raw prompts and completions are intentional
    experimental records.
``meta.json``
    Immutable run-level provenance, written exactly once by the runner after it has
    resolved its complete condition/configuration.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any


@dataclasses.dataclass
class TickRecord:
    run_id: str
    tick: int
    agent_id: str
    alive: bool
    drives: dict[str, float]
    position: str
    chosen_action: str
    perceived: str          # exactly what the agent perceived (post-salience)
    self_model_version: int
    cause_of_death: str | None = None
    # The fields below were added for the simultaneous-proposal T5 design. They
    # intentionally default to None so old analysis scripts and hand-built records
    # remain valid.
    proposal_index: int | None = None
    resolution_index: int | None = None
    proposed_action: str | None = None
    # ``drives`` and ``position`` above are retained as the post-resolution
    # state for compatibility with the original log schema.  The fields below
    # preserve the state that actually informed the decision.  Without this
    # distinction, an analysis can accidentally call a successful consume
    # "discretionary" because it examines the lower *post-consume* drive.
    pre_action_drives: dict[str, float] | None = None
    pre_action_position: str | None = None
    # JSON-safe, internal drive-name -> count snapshot.  This is deliberately
    # separate from perceived text: it records the available private stock,
    # while ``perceived`` remains the exact salience-filtered context shown to
    # the agent.
    pre_action_inventory: dict[str, int] | None = None
    # Structured, pre-resolution opportunity set (resources, co-located
    # agents, reachable locations, and action-relevant affordances).  It lets
    # downstream analyses separate a choice from an unavailable opportunity.
    pre_action_opportunities: dict[str, Any] | None = None
    # ``None`` means a custom/stub policy did not expose parser diagnostics;
    # old records therefore remain analyzable.  LLM policies record ``ok`` or
    # an explicit fallback reason such as ``no_legal_action``.
    parse_status: str | None = None
    # Whether the submitted proposal was syntactically/parse-valid before
    # resolution.  A malformed completion that falls back to ``wait`` is
    # False, so it cannot be mistaken for an intentional wait.
    proposal_valid: bool | None = None
    # ``outcome`` is the human-readable realized outcome.  These structured
    # fields make success/failure usable without reverse-parsing that prose.
    outcome_status: str | None = None
    action_succeeded: bool | None = None
    outcome: str | None = None
    reflection_reason: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


@dataclasses.dataclass
class EventRecord:
    run_id: str
    tick: int
    kind: str               # "death" | "shock" | "rescue" | "resource_depleted" | ...
    agent_id: str | None
    detail: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


@dataclasses.dataclass
class CompletionRecord:
    """One model request plus its parsed result.

    ``phase`` is normally ``"action"`` or ``"reflection"``. ``parse_status`` is
    deliberately a free string: action policies can distinguish a legal parse from a
    fallback, while reflection calls can record ``"not_applicable"``. The logger
    fills ``run_id`` so callers cannot accidentally attach a completion to the wrong
    run directory.
    """

    phase: str
    tick: int | None
    agent_id: str | None
    system_prompt: str | None
    user_prompt: str
    raw_completion: str | None
    parsed_output: str | None
    parse_status: str
    request_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    response_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    run_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


class RunLogger:
    """Write immutable, structured logs for one run into ``<root>/<run_id>/``.

    A run directory is created with ``mkdir`` (rather than ``exist_ok=True``), which
    makes a collision an immediate failure instead of silently appending a new
    condition to old data. Call :meth:`log_meta` exactly once after the runner has
    assembled its full resolved configuration.
    """

    def __init__(self, run_id: str, root: str = "runs"):
        self.run_id = run_id
        self.dir = os.path.join(root, run_id)

        # Make any parent hierarchy first, then create the leaf with one atomic OS
        # operation. This supports condition/seed subdirectories while still
        # preventing a racing process from reusing the same actual run directory.
        parent = os.path.dirname(os.path.normpath(self.dir)) or os.curdir
        os.makedirs(parent, exist_ok=True)
        try:
            os.mkdir(self.dir)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Run directory already exists and is immutable: {self.dir!r}. "
                "Choose a fresh condition-specific run id."
            ) from exc

        # The directory is newly created, so exclusive stream creation gives us an
        # additional guard against accidental reuse or an unexpected pre-populated
        # filesystem.
        self._self_models = open(
            os.path.join(self.dir, "self_models.jsonl"), "x", encoding="utf-8")
        self._ticks = open(
            os.path.join(self.dir, "ticks.jsonl"), "x", encoding="utf-8")
        self._events = open(
            os.path.join(self.dir, "events.jsonl"), "x", encoding="utf-8")
        self._completions = open(
            os.path.join(self.dir, "completions.jsonl"), "x", encoding="utf-8")
        self._meta_written = False

    def log_meta(self, meta: dict[str, Any]) -> None:
        """Persist complete run metadata exactly once.

        The constructor intentionally does not write a partial ``meta.json``. In
        particular, a runner must be able to include its resolved config, prompts,
        model/decoding settings, label map, and seed in the sole immutable record.
        """
        if self._meta_written:
            raise RuntimeError("Run metadata is immutable and has already been written")

        payload = dict(meta)
        supplied_run_id = payload.get("run_id", self.run_id)
        if supplied_run_id != self.run_id:
            raise ValueError(
                f"metadata run_id {supplied_run_id!r} does not match logger run_id "
                f"{self.run_id!r}"
            )
        payload["run_id"] = self.run_id
        payload.setdefault("created", time.time())

        # Serialize before making the file so a non-serializable config does not
        # leave behind a half-written, permanently immutable metadata file.
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        path = os.path.join(self.dir, "meta.json")
        try:
            with open(path, "x", encoding="utf-8") as f:
                f.write(serialized)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError as exc:
            # This also protects against an external process trying to inject or
            # replace provenance after the run directory was created.
            raise RuntimeError("Run metadata is immutable and already exists") from exc
        self._meta_written = True

    def log_self_model(self, agent_id: str, record: dict[str, Any]) -> None:
        record = {**record, "agent_id": agent_id, "run_id": self.run_id}
        self._self_models.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._self_models.flush()

    def log_tick(self, record: TickRecord) -> None:
        self._ticks.write(record.to_json() + "\n")
        self._ticks.flush()

    def log_event(self, record: EventRecord) -> None:
        self._events.write(record.to_json() + "\n")
        self._events.flush()

    def log_completion(self, record: CompletionRecord | dict[str, Any]) -> None:
        """Append one prompt/completion provenance record.

        A mapping is accepted as a convenience for runners that already assemble a
        JSON-ready record. ``run_id`` is always set from this logger, rather than
        trusting a caller-provided value.
        """
        if isinstance(record, CompletionRecord):
            payload = dataclasses.asdict(record)
        else:
            payload = dict(record)
        payload["run_id"] = self.run_id
        self._completions.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._completions.flush()

    def close(self) -> None:
        for f in (self._self_models, self._ticks, self._events, self._completions):
            f.close()
