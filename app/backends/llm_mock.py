"""Deterministic mock LLM.

Streams a short reply token-by-token (with a tiny delay so streaming/TTFT is
observable) and always emits an [EMOTION:xxx] tag so the emotion contract can be
exercised end-to-end without a model.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

from app.backends.base import LLMBackend
from app.schemas import ChatMessage


class MockLLM(LLMBackend):
    name = "mock"

    async def load(self) -> None:
        await asyncio.sleep(0)

    async def stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        reply = (
            f"[EMOTION:happy] I heard you say: {last_user.strip() or 'nothing'}. "
            "This is the Brutus brain node mock backend speaking."
        )
        words = reply.split(" ")
        limit = max_tokens if max_tokens and max_tokens > 0 else len(words)
        for i, word in enumerate(words[:limit]):
            await asyncio.sleep(0.01)
            yield word if i == 0 else " " + word
