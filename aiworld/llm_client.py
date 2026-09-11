"""A single, provider-agnostic LLM client.

Everything downstream calls `LLMClient.complete(prompt) -> str`. The client speaks the
OpenAI-compatible /v1/chat/completions protocol, which is exposed by ALL of:
  - Ollama            (http://localhost:11434/v1)
  - vLLM              (http://localhost:8000/v1)
  - LM Studio         (http://localhost:1234/v1)
  - the hosted APIs   (Anthropic via its own SDK; OpenAI/together/etc. directly)

So the same experiment runs on your local 5090 or a cloud model by changing base_url.
This is also what the cross-model confound check (DESIGN.md §8) needs: swap the model
id, rerun, compare.

No heavy SDK dependency — just `requests`. Anthropic is handled as a special case
because its wire format differs; everything else is OpenAI-compatible.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any

import requests


@dataclasses.dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"   # Ollama default
    model: str = "qwen3.6:35b"
    api_key: str = "not-needed-for-local"          # local servers ignore this
    temperature: float = 0.8      # some stochasticity: identical agents must be able
                                  # to diverge. Too low collapses them; too high is noise.
    max_tokens: int = 300
    timeout: int = 120
    provider: str = "openai"      # "openai" (covers local) | "anthropic"
    # Retry/backoff for flaky local servers under concurrent load.
    max_retries: int = 3
    retry_backoff: float = 2.0


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.n_calls = 0
        self.total_latency = 0.0

    def complete(self, prompt: str, system: str | None = None) -> str:
        if self.config.provider == "anthropic":
            return self._complete_anthropic(prompt, system)
        return self._complete_openai(prompt, system)

    # ---- OpenAI-compatible (Ollama, vLLM, LM Studio, OpenAI, ...) ----
    def _complete_openai(self, prompt: str, system: str | None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key != "not-needed-for-local":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        data = self._post(url, payload, headers)
        return data["choices"][0]["message"]["content"]

    # ---- Anthropic (different wire format) ----
    def _complete_anthropic(self, prompt: str, system: str | None) -> str:
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
        data = self._post(url, payload, headers)
        # Anthropic returns a list of content blocks.
        return "".join(b.get("text", "") for b in data.get("content", []))

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        last_err = None
        for attempt in range(self.config.max_retries):
            try:
                t0 = time.time()
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=self.config.timeout)
                resp.raise_for_status()
                self.n_calls += 1
                self.total_latency += time.time() - t0
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.config.retry_backoff * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} "
                           f"retries: {last_err}")

    def stats(self) -> dict[str, float]:
        avg = self.total_latency / self.n_calls if self.n_calls else 0.0
        return {"n_calls": self.n_calls,
                "total_latency_s": round(self.total_latency, 2),
                "avg_latency_s": round(avg, 3)}


def config_from_env() -> LLMConfig:
    """Build config from environment variables so you never hard-code endpoints.

    LOCAL (Ollama):   leave AIWORLD_PROVIDER unset or =openai, base_url default.
    HOSTED Anthropic: AIWORLD_PROVIDER=anthropic, AIWORLD_BASE_URL=https://api.anthropic.com/v1,
                      AIWORLD_MODEL=claude-..., ANTHROPIC_API_KEY=...
    """
    provider = os.getenv("AIWORLD_PROVIDER", "openai")
    default_base = ("http://localhost:11434/v1" if provider == "openai"
                    else "https://api.anthropic.com/v1")
    key = (os.getenv("ANTHROPIC_API_KEY") if provider == "anthropic"
           else os.getenv("AIWORLD_API_KEY", "not-needed-for-local"))
    return LLMConfig(
        base_url=os.getenv("AIWORLD_BASE_URL", default_base),
        model=os.getenv("AIWORLD_MODEL", "qwen3.6:35b"),
        api_key=key or "not-needed-for-local",
        temperature=float(os.getenv("AIWORLD_TEMPERATURE", "0.8")),
        provider=provider,
    )
