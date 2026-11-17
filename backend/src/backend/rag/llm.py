"""Minimal LLM clients (stdlib only). Provider chosen with LLM_PROVIDER = anthropic | openai.

`openai` means any OpenAI-compatible /chat/completions server (set LLM_BASE_URL and LLM_MODEL).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Protocol

from ..config import Settings

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
Post = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


class LLMError(RuntimeError):
    """The LLM call failed (network, HTTP error, or an empty/odd response)."""


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


def _urllib_post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"LLM API returned HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise LLMError(f"LLM API call failed: {e}") from e


class AnthropicClient:
    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = 800,
                 timeout: float = 60, base_url: str = "", post: Post = _urllib_post):
        self.api_key, self.model, self.max_tokens, self.timeout = api_key, model, max_tokens, timeout
        self.url = (base_url.rstrip("/") or "https://api.anthropic.com") + "/v1/messages"
        self._post = post

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            self.url,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            self.timeout,
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        if not text:
            raise LLMError(f"Empty response from Anthropic API: {str(data)[:300]}")
        return text


class OpenAICompatClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 800, timeout: float = 60,
                 base_url: str = "", post: Post = _urllib_post):
        self.api_key, self.model, self.max_tokens, self.timeout = api_key, model, max_tokens, timeout
        self.url = (base_url.rstrip("/") or "https://api.openai.com/v1") + "/chat/completions"
        self._post = post

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            self.url,
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            self.timeout,
        )
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response from LLM API: {str(data)[:300]}") from e
        if not text:
            raise LLMError("Empty response from LLM API")
        return text


def build_llm_client(cfg: Settings) -> Optional[LLMClient]:
    """None when LLM_API_KEY is not set -> the API runs in retrieval-only mode."""
    if not cfg.llm_api_key:
        return None
    if cfg.llm_provider == "anthropic":
        return AnthropicClient(cfg.llm_api_key, cfg.llm_model or DEFAULT_ANTHROPIC_MODEL,
                               cfg.llm_max_tokens, cfg.llm_timeout, cfg.llm_base_url)
    if cfg.llm_provider == "openai":
        if not cfg.llm_model:
            raise ValueError("LLM_MODEL must be set when LLM_PROVIDER=openai")
        return OpenAICompatClient(cfg.llm_api_key, cfg.llm_model, cfg.llm_max_tokens,
                                  cfg.llm_timeout, cfg.llm_base_url)
    raise ValueError(f"Unknown LLM_PROVIDER {cfg.llm_provider!r} (use 'anthropic' or 'openai')")
