from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import agents
from openai import AsyncOpenAI


def openrouter_async_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter-backed OpenAI client.")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def openrouter_async_model(model: str = "openai/gpt-4.1-mini") -> Callable[..., Any]:
    """
    Return an async OpenAI-only call wrapper pinned to `model`.

    Usage:
        create = openrouter_async_model("openai/gpt-4.1-mini")
        resp = await create(messages=[...], temperature=0.2)
    """
    client = openrouter_async_client()

    async def create_chat_completion(**kwargs: Any) -> Any:
        return await client.chat.completions.create(model=model, **kwargs)

    return create_chat_completion


def openrouter_async_agents_client() -> AsyncOpenAI:
    # OpenAI Agents SDK enables tracing by default and logs
    # "OPENAI_API_KEY is not set, skipping trace export" when only OpenRouter is configured.
    # Disable tracing explicitly to keep logs clean in OpenRouter-only mode.
    agents.set_tracing_disabled(True)
    client = openrouter_async_client()
    agents.set_default_openai_client(client, use_for_tracing=False)
    return client


def openrouter_async_agents_model(model: str = "openai/gpt-4.1-mini"):
    return agents.OpenAIChatCompletionsModel(
        model=model,
        openai_client=openrouter_async_agents_client(),
    )
