"""AnythingLLM proxy backend (the "fast path").

Streams from a running AnythingLLM instance via its OpenAI-compatible endpoint.
Lets the Brain Node expose the same /v1/chat contract in hour one -- with zero
model code -- while the native genai backend is brought up alongside it.

Enable with BRUTUS_LLM_BACKEND=anythingllm and set the base URL + API key.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, List

import httpx

from app.backends.base import LLMBackend
from app.schemas import ChatMessage
from config import get_settings


class AnythingLLMProxy(LLMBackend):
    name = "anythingllm"

    async def load(self) -> None:
        # Connectivity is validated lazily on first request.
        return None

    async def stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:
        settings = get_settings()
        url = settings.anythingllm_base_url.rstrip("/") + "/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.anythingllm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.anythingllm_workspace,
            "stream": True,
            "temperature": temperature,
            "messages": [m.model_dump() for m in messages],
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
