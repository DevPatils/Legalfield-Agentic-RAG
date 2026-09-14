"""LLM access for the agent nodes.

Every call returns a validated Pydantic model -- there is no free-text parsing anywhere
in the graph. Latency and token usage are captured per call because Architecture.md §10
stores a ``latency_ms`` breakdown on every trace.

All calls happen server-side; the frontend never sees a key (Architecture.md §8).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

from ..config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMCall:
    """One completed call, for the trace."""

    node: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class UsageLog:
    """Accumulates per-query cost/latency so a trace can show where time went."""

    calls: list[LLMCall] = field(default_factory=list)

    def record(self, call: LLMCall) -> None:
        self.calls.append(call)

    def latency_by_node(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for call in self.calls:
            out[call.node] = out.get(call.node, 0) + call.latency_ms
        return out

    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(c.input_tokens for c in self.calls),
            "output": sum(c.output_tokens for c in self.calls),
        }


def _require_parsed(parsed: T | None, node: str, stop_reason: str = "") -> T:
    """Turn an unparseable response into an exception instead of a ``None``.

    ``messages.parse`` returns ``parsed_output=None`` rather than raising when the model
    does not produce something matching the schema -- most often because it hit
    ``max_tokens`` part-way through the structure. Every node wraps its call in
    try/except and degrades to a defensible default, but that guard only catches
    raises: a None slipped past it and blew up on the next attribute access, outside
    the guard, killing the whole run and returning no answer at all. Architecture.md §7
    requires the graph to degrade rather than fail closed, and this is the line that
    decides which of the two happens.
    """
    if parsed is None:
        detail = f" (stop_reason={stop_reason})" if stop_reason else ""
        raise RuntimeError(f"{node or 'llm'}: model returned no parseable output{detail}")
    return parsed


class LLMClient(Protocol):
    def parse(
        self,
        system: str,
        user: str,
        schema: type[T],
        node: str = "",
        fast: bool = False,
        max_tokens: int = 4096,
    ) -> T: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, fast_model: str, usage: UsageLog) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.fast_model = fast_model
        self.usage = usage

    def parse(
        self,
        system: str,
        user: str,
        schema: type[T],
        node: str = "",
        fast: bool = False,
        max_tokens: int = 4096,
    ) -> T:
        model = self.fast_model if fast else self.model
        started = time.perf_counter()
        response = self.client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        self.usage.record(
            LLMCall(
                node=node,
                model=model,
                latency_ms=elapsed,
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
            )
        )
        return _require_parsed(response.parsed_output, node, getattr(response, "stop_reason", ""))


class OpenAILLM:
    def __init__(self, api_key: str, model: str, fast_model: str, usage: UsageLog) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fast_model = fast_model
        self.usage = usage

    def parse(
        self,
        system: str,
        user: str,
        schema: type[T],
        node: str = "",
        fast: bool = False,
        max_tokens: int = 4096,
    ) -> T:
        model = self.fast_model if fast else self.model
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": False,
                },
            },
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        usage = response.usage
        self.usage.record(
            LLMCall(
                node=node,
                model=model,
                latency_ms=elapsed,
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            )
        )
        return schema.model_validate(json.loads(response.choices[0].message.content or "{}"))


def get_llm(usage: UsageLog, settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is unset.")
        return AnthropicLLM(
            settings.anthropic_api_key, settings.llm_model, settings.llm_model_fast, usage
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is unset.")
        return OpenAILLM(
            settings.openai_api_key, settings.llm_model, settings.llm_model_fast, usage
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
