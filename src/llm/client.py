"""DeepSeek client with JSON-mode helpers.

Important behaviour of `deepseek-v4-flash`: it is a *reasoning* model. It emits
`reasoning_content` first and the real answer afterwards, and reasoning tokens
are billed against `max_tokens`. With a small budget the response comes back
with an EMPTY content string and `finish_reason: length`.

This client therefore:
  * uses a generous max_tokens floor (see `REASONING_TOKEN_FLOOR`),
  * retries once with a larger budget when content is empty,
  * extracts JSON defensively, since models still occasionally wrap it in
    markdown fences despite `response_format`.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from ..config import LLMConfig, get_settings

logger = logging.getLogger(__name__)

#: Below this, a reasoning model can spend the whole budget thinking.
REASONING_TOKEN_FLOOR = 1024

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLMError(RuntimeError):
    """The model could not be reached or produced unusable output."""


class DeepSeekClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or get_settings().llm
        if not self.config.api_key:
            raise LLMError("LLM_API_KEY is not configured")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        retries: int = 2,
    ) -> str:
        budget = max(REASONING_TOKEN_FLOOR, max_tokens or self.config.max_tokens)
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": budget,
                "temperature": self.config.temperature if temperature is None else temperature,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            try:
                content = self._post(payload)
                if content.strip():
                    return content
                # Empty content => reasoning consumed the whole budget.
                logger.warning(
                    "Empty content from %s (max_tokens=%s); retrying with a larger budget",
                    self.config.model, budget,
                )
                budget = min(budget * 2, 8192)
            except Exception as exc:
                last_error = exc
                logger.warning("LLM call failed (attempt %s): %s", attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))

        raise LLMError(f"LLM produced no usable content after {retries + 1} attempts: {last_error}")

    def _post(self, payload: Dict[str, Any]) -> str:
        response = requests.post(
            self.config.chat_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout,
        )
        if response.status_code >= 400:
            raise LLMError(f"LLM HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if not content and choice.get("finish_reason") == "length":
            logger.debug("Response truncated: reasoning consumed max_tokens=%s", payload["max_tokens"])
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Chat, then parse the reply as a JSON object.

        Raises LLMError if the model does not return parseable JSON - callers
        are expected to degrade rather than trust a stringly-typed response.
        """
        raw = self.chat(messages, max_tokens=max_tokens, temperature=temperature, json_mode=True)
        parsed = extract_json(raw)
        if parsed is None:
            raise LLMError(f"Model did not return valid JSON. Raw output: {raw[:300]}")
        return parsed


def extract_json(raw: str) -> Optional[Any]:
    """Pull a JSON value out of a model response.

    Handles fenced blocks, leading prose, and trailing commentary.
    """
    if not raw:
        return None
    text = raw.strip()

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Scan for the first balanced object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
    return None


_client: DeepSeekClient | None = None


def get_llm_client() -> DeepSeekClient:
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client
