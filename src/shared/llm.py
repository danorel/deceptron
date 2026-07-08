"""LLM client abstraction — Anthropic and OpenAI-compatible (vLLM / Sesterce) backends.

Internal message format is Anthropic-style content blocks so that trajectory
storage stays provider-agnostic. Adapters convert to/from the provider wire format.

Environment variables:
    LLM_PROVIDER    "anthropic" (default) | "openai"
    LLM_BASE_URL    Required for "openai" provider — vLLM / Sesterce endpoint
    ANTHROPIC_API_KEY | OPENAI_API_KEY
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

RETRY_DELAYS = [15, 30, 60]


@dataclass
class LLMResponse:
    """Provider-agnostic response. content = Anthropic-style blocks (list[dict])."""
    content: list[dict]   # [{"type":"text","text":"..."}, {"type":"tool_use","id":...}]
    stop_reason: str      # "end_turn" | "tool_use"


class LLMClient(ABC):
    """Abstract LLM client. All callers use internal Anthropic-style message format."""

    @abstractmethod
    def _call(
        self,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
    ) -> LLMResponse: ...

    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call with automatic retry on rate-limit errors."""
        for attempt, delay in enumerate([0] + RETRY_DELAYS, 1):
            if delay:
                print(f"    rate limit — waiting {delay}s (attempt {attempt}/{len(RETRY_DELAYS)+1})")
                time.sleep(delay)
            try:
                return self._call(model, system, messages, tools or [], max_tokens)
            except Exception as e:
                code = getattr(e, "status_code", None)
                if code is None:
                    resp = getattr(e, "response", None)
                    code = getattr(resp, "status_code", None)
                if code in (429, 529) and attempt <= len(RETRY_DELAYS):
                    continue
                raise


class AnthropicLLMClient(LLMClient):
    """Wraps the official Anthropic SDK. Adds prompt caching on system block."""

    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _call(self, model, system, messages, tools, max_tokens) -> LLMResponse:
        system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_block,
            tools=tools or [],
            messages=messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        content = [b.model_dump() for b in response.content]
        stop_reason = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
        return LLMResponse(content=content, stop_reason=stop_reason)


class OpenAILLMClient(LLMClient):
    """Wraps any OpenAI-compatible endpoint (vLLM, Sesterce, etc.)."""

    def __init__(self, api_key: str, base_url: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ── format converters ──────────────────────────────────────────────────────

    @staticmethod
    def _to_oai_messages(system: str, messages: list[dict]) -> list[dict]:
        """Anthropic-style messages → OpenAI messages list (system first)."""
        result: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if isinstance(content, str):
                    result.append({"role": "user", "content": content})
                    continue
                tool_results = [b for b in content if b.get("type") == "tool_result"]
                text_blocks  = [b for b in content if b.get("type") == "text"]
                if tool_results:
                    for tr in tool_results:
                        result.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": tr.get("content", ""),
                        })
                else:
                    text = "\n".join(b.get("text", "") for b in text_blocks)
                    result.append({"role": "user", "content": text})

            elif role == "assistant":
                if isinstance(content, str):
                    result.append({"role": "assistant", "content": content})
                    continue
                text = ""
                tool_calls = []
                for b in content:
                    if b.get("type") == "text":
                        text += b.get("text", "")
                    elif b.get("type") == "tool_use":
                        tool_calls.append({
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        })
                out: dict = {"role": "assistant", "content": text or None}
                if tool_calls:
                    out["tool_calls"] = tool_calls
                result.append(out)

        return result

    @staticmethod
    def _to_oai_tools(tools: list[dict]) -> list[dict]:
        """Anthropic tool defs (input_schema) → OpenAI function tool defs (parameters)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _from_oai_response(response) -> LLMResponse:
        """OpenAI response → internal Anthropic-style content blocks."""
        choice = response.choices[0]
        msg = choice.message
        content: list[dict] = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        for tc in (msg.tool_calls or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": arguments,
            })
        stop_reason = "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"
        return LLMResponse(content=content, stop_reason=stop_reason)

    def _call(self, model, system, messages, tools, max_tokens) -> LLMResponse:
        oai_messages = self._to_oai_messages(system, messages)
        kwargs: dict = dict(model=model, messages=oai_messages, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = self._to_oai_tools(tools)
        response = self._client.chat.completions.create(**kwargs)
        return self._from_oai_response(response)


def resolve_model(default: str) -> str:
    """Return LLM_MODEL env var if set, else the module-level default."""
    return os.environ.get("LLM_MODEL", default)


def make_client(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Factory — reads LLM_PROVIDER / LLM_BASE_URL / *_API_KEY from env if not provided."""
    provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        url = base_url or os.environ.get("LLM_BASE_URL", "")
        if not url:
            raise ValueError("LLM_BASE_URL must be set for openai provider")
        return OpenAILLMClient(api_key=key, base_url=url)
    else:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        return AnthropicLLMClient(api_key=key)


# ── backward-compat shim for analyze.py (still uses raw anthropic client) ──────

def create_with_retry(client, **kwargs):
    """Legacy wrapper. Use LLMClient.chat() for new code."""
    import anthropic
    for attempt, delay in enumerate([0] + RETRY_DELAYS, 1):
        if delay:
            print(f"    rate limit — waiting {delay}s (attempt {attempt}/{len(RETRY_DELAYS)+1})")
            time.sleep(delay)
        try:
            return client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt <= len(RETRY_DELAYS):
                continue
            raise
