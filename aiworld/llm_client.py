"""A small provider-agnostic LLM client with reproducibility provenance.

Everything downstream calls ``LLMClient.complete(prompt) -> str``. The client speaks
the OpenAI-compatible ``/v1/chat/completions`` protocol exposed by Ollama, vLLM, LM
Studio, OpenAI, and similar servers. Anthropic is handled separately because its
wire format differs.

The return value stays deliberately simple, but callers can obtain safe request and
response metadata after every call via :meth:`request_metadata` and
:meth:`response_metadata`. Those dictionaries never contain API keys or headers and
are intended for the run's ``completions.jsonl`` provenance stream.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


@dataclasses.dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"   # Ollama default
    model: str = "huihui_ai/mistral-small-abliterated:24b"
    api_key: str = "not-needed-for-local"          # local servers ignore this
    temperature: float = 0.8      # some stochasticity: identical agents must be able
                                  # to diverge. Too low collapses them; too high is noise.
    max_tokens: int = 300
    timeout: int = 120
    provider: str = "openai"      # "openai" (covers local) | "anthropic"
    # Retry/backoff for flaky local servers under concurrent load.
    max_retries: int = 3
    retry_backoff: float = 2.0
    # OpenAI-compatible servers generally accept ``seed``. It is intentionally
    # optional because not all local servers implement it. Anthropic's Messages API
    # has no equivalent request field, so it is recorded as unsupported rather than
    # silently pretending the request was seeded.
    sampling_seed: int | None = None
    # Ollama-specific context-window cap, sent as {"options": {"num_ctx": N}} on
    # the OpenAI-compatible endpoint (Ollama passes this through; other
    # OpenAI-compatible servers generally ignore unknown top-level fields).
    # None (default) omits it entirely -- unchanged behavior, whatever the
    # server's own default context length is. This does NOT change the
    # dominant VRAM cost (the model's own weights); it only bounds the
    # KV-cache allocation, which scales with context length. Set well above
    # this project's actual prompts (~1200 chars / ~300 tokens action
    # context, capped max_tokens=300 completion) rather than at a server
    # default that may be far larger than anything actually sent.
    num_ctx: int | None = None


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.n_calls = 0
        self.total_latency = 0.0
        # Public read-only-by-convention snapshots for lightweight integrations.
        # The accessor methods return copies so loggers cannot mutate client state.
        self.last_request_metadata: dict[str, Any] = self._base_request_metadata()
        self.last_response_metadata: dict[str, Any] = {}
        self._last_transport_metadata: dict[str, Any] = {}

    def complete(self, prompt: str, system: str | None = None,
                seed: int | None = None) -> str:
        """Request one completion.

        ``seed`` overrides ``self.config.sampling_seed`` for this call only,
        without mutating shared config -- the mechanism that lets a caller
        give every repeated draw from one frozen context its own distinct
        seed (aiworld.seed_plan.SeedPlan) instead of a decoder-null collapsing
        onto one reused value. Omit it to keep the previous config-default
        behavior unchanged.
        """
        if self.config.provider == "anthropic":
            return self._complete_anthropic(prompt, system, seed)
        return self._complete_openai(prompt, system, seed)

    def request_metadata(self) -> dict[str, Any]:
        """Return safe metadata for the most recent request (never credentials).

        This includes resolved model/decoding settings and whether a sampling seed was
        actually sent. It intentionally omits prompt text: the caller owns that
        provenance and can log it alongside phase/tick/agent context.
        """
        return dict(self.last_request_metadata)

    def response_metadata(self) -> dict[str, Any]:
        """Return safe model/fingerprint/usage metadata from the latest response."""
        return dict(self.last_response_metadata)

    # ---- OpenAI-compatible (Ollama, vLLM, LM Studio, OpenAI, ...) ----
    def _complete_openai(self, prompt: str, system: str | None,
                         seed: int | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        effective_seed = seed if seed is not None else self.config.sampling_seed
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if effective_seed is not None:
            payload["seed"] = effective_seed
        if self.config.num_ctx is not None:
            payload["options"] = {"num_ctx": self.config.num_ctx}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key != "not-needed-for-local":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        self._start_request_metadata(
            endpoint="/chat/completions", seed_sent=(effective_seed is not None),
            effective_seed=effective_seed,
            seed_source=self._seed_source(seed))
        data = self._post(url, payload, headers)
        self._capture_openai_response_metadata(data)
        return data["choices"][0]["message"]["content"]

    # ---- Anthropic (different wire format) ----
    def _complete_anthropic(self, prompt: str, system: str | None,
                            seed: int | None) -> str:
        url = self.config.base_url.rstrip("/") + "/messages"
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        effective_seed = seed if seed is not None else self.config.sampling_seed
        # Anthropic does not currently offer the OpenAI-compatible seed request
        # parameter. Preserve the requested value in metadata, but do not inject a
        # made-up field into its API payload.
        self._start_request_metadata(
            endpoint="/messages", seed_sent=False, effective_seed=effective_seed,
            seed_source=self._seed_source(seed))
        data = self._post(url, payload, headers)
        self._capture_anthropic_response_metadata(data)
        # Anthropic returns a list of content blocks.
        return "".join(b.get("text", "") for b in data.get("content", []))

    def _seed_source(self, call_seed: int | None) -> str | None:
        """Which value actually determined this call's effective seed."""
        if call_seed is not None:
            return "call"
        if self.config.sampling_seed is not None:
            return "config"
        return None

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        last_err = None
        for attempt in range(self.config.max_retries):
            try:
                t0 = time.time()
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=self.config.timeout)
                resp.raise_for_status()
                latency = time.time() - t0
                self.n_calls += 1
                self.total_latency += latency
                self._last_transport_metadata = {
                    "attempts": attempt + 1,
                    "latency_s": round(latency, 6),
                    "http_status": getattr(resp, "status_code", None),
                }
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.config.retry_backoff * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} "
                           f"retries: {last_err}")

    def _base_request_metadata(self) -> dict[str, Any]:
        """Configuration provenance safe to persist in a public run directory."""
        seed_supported = self.config.provider != "anthropic"
        return {
            "provider": self.config.provider,
            "base_url": self._safe_base_url(self.config.base_url),
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout_s": self.config.timeout,
            "max_retries": self.config.max_retries,
            "retry_backoff_s": self.config.retry_backoff,
            "sampling_seed": self.config.sampling_seed,
            "sampling_seed_supported": seed_supported,
            "sampling_seed_sent": False,
            "num_ctx": self.config.num_ctx,
        }

    def _start_request_metadata(self, endpoint: str, seed_sent: bool,
                                effective_seed: int | None = None,
                                seed_source: str | None = None) -> None:
        metadata = self._base_request_metadata()
        metadata["endpoint"] = endpoint
        metadata["sampling_seed_sent"] = seed_sent
        # The seed that actually determined this call, and whether it came
        # from an explicit per-call override (e.g. SeedPlan) or the client's
        # static config default -- distinguishing these is what lets a static
        # sampling null prove it used a distinct seed per draw rather than
        # silently repeating one value across every call.
        metadata["effective_sampling_seed"] = effective_seed
        metadata["sampling_seed_source"] = seed_source
        self.last_request_metadata = metadata
        # A failed request must not leave a previous call's fingerprint attached to
        # the new completion record.
        self.last_response_metadata = {}
        self._last_transport_metadata = {}

    def _capture_openai_response_metadata(self, data: dict[str, Any]) -> None:
        choices = data.get("choices") or []
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        metadata: dict[str, Any] = {
            "provider": self.config.provider,
            "response_id": data.get("id"),
            "model": data.get("model", self.config.model),
            "system_fingerprint": data.get("system_fingerprint"),
            "created": data.get("created"),
            "usage": data.get("usage"),
            "finish_reason": first_choice.get("finish_reason"),
            **self._last_transport_metadata,
        }
        self.last_response_metadata = self._without_none(metadata)

    def _capture_anthropic_response_metadata(self, data: dict[str, Any]) -> None:
        metadata: dict[str, Any] = {
            "provider": self.config.provider,
            "response_id": data.get("id"),
            "model": data.get("model", self.config.model),
            # Some gateways may expose this even though the native API normally
            # does not; retaining it makes fingerprints comparable when available.
            "system_fingerprint": data.get("system_fingerprint"),
            "type": data.get("type"),
            "stop_reason": data.get("stop_reason"),
            "stop_sequence": data.get("stop_sequence"),
            "usage": data.get("usage"),
            **self._last_transport_metadata,
        }
        self.last_response_metadata = self._without_none(metadata)

    @staticmethod
    def _without_none(values: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def _safe_base_url(url: str) -> str:
        """Strip credentials and query/fragment data before persisting a URL."""
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.netloc:
            # A relative local endpoint carries no credentials in the normal client
            # configuration. Retain it verbatim rather than producing a malformed
            # URL in provenance.
            return url
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))

    def stats(self) -> dict[str, float]:
        avg = self.total_latency / self.n_calls if self.n_calls else 0.0
        return {"n_calls": self.n_calls,
                "total_latency_s": round(self.total_latency, 2),
                "avg_latency_s": round(avg, 3)}


def config_from_env() -> LLMConfig:
    """Build config from environment variables so you never hard-code endpoints.

    LOCAL (Ollama): leave AIWORLD_PROVIDER unset or =openai, base_url default.
    HOSTED Anthropic: AIWORLD_PROVIDER=anthropic, AIWORLD_BASE_URL=https://api.anthropic.com/v1,
                      AIWORLD_MODEL=claude-..., ANTHROPIC_API_KEY=...

    ``AIWORLD_SAMPLING_SEED`` is forwarded to OpenAI-compatible backends only. It is
    kept in provenance but deliberately not sent to Anthropic, whose Messages API
    does not support that field.
    """
    provider = os.getenv("AIWORLD_PROVIDER", "openai")
    default_base = ("http://localhost:11434/v1" if provider == "openai"
                    else "https://api.anthropic.com/v1")
    key = (os.getenv("ANTHROPIC_API_KEY") if provider == "anthropic"
           else os.getenv("AIWORLD_API_KEY", "not-needed-for-local"))
    raw_sampling_seed = os.getenv("AIWORLD_SAMPLING_SEED")
    sampling_seed = (int(raw_sampling_seed)
                     if raw_sampling_seed not in (None, "") else None)
    raw_num_ctx = os.getenv("AIWORLD_NUM_CTX")
    num_ctx = int(raw_num_ctx) if raw_num_ctx not in (None, "") else None
    return LLMConfig(
        base_url=os.getenv("AIWORLD_BASE_URL", default_base),
        model=os.getenv("AIWORLD_MODEL", "huihui_ai/mistral-small-abliterated:24b"),
        api_key=key or "not-needed-for-local",
        temperature=float(os.getenv("AIWORLD_TEMPERATURE", "0.8")),
        provider=provider,
        sampling_seed=sampling_seed,
        num_ctx=num_ctx,
    )
