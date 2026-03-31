# Author: Max Vogeltanz, University of Graz, 2026

# OpenAI Chat Completions API client. Also used for OpenAI-compatible servers (e.g. institutional vLLM).

import os
from collections.abc import Mapping
from typing import Any

from openai import OpenAI

from .base import GenResult, Usage


def _assistant_text(message: Any) -> str:
    """Normalize assistant text from Chat Completions (incl. vLLM / Qwen reasoning channels)."""
    d = message.model_dump(mode="python")
    c = d.get("content")
    if isinstance(c, str) and c.strip():
        return c.strip()
    if isinstance(c, list):
        parts: list[str] = []
        for part in c:
            if isinstance(part, Mapping):
                if part.get("type") == "text":
                    t = part.get("text")
                    if t:
                        parts.append(str(t))
                else:
                    t = part.get("text") or part.get("content")
                    if t:
                        parts.append(str(t))
            else:
                t = getattr(part, "text", None)
                if t:
                    parts.append(str(t))
        joined = "".join(parts).strip()
        if joined:
            return joined
    for key in ("reasoning_content", "reasoning", "thinking"):
        extra = d.get(key)
        if isinstance(extra, str) and extra.strip():
            return extra.strip()
    mex = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning", "thinking"):
        v = mex.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        # Local OpenAI-compatible servers (e.g. vLLM) often do not require real keys.
        # The OpenAI SDK still expects a non-empty api_key value, so we provide a safe placeholder.
        api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-local"
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self._extra_body = extra_body

    def generate(self, *, system: str, user: str, model: str, max_tokens: int, temperature: float):
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        text = _assistant_text(msg)
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return GenResult(text=text, usage=usage)
