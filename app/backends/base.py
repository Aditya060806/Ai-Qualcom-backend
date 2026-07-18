"""Abstract backend interfaces.

Every backend exposes an async ``load()`` (called once at boot) and its work
method. Real backends push blocking model calls onto a thread with
``asyncio.to_thread`` so the event loop stays responsive while streaming.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional, Tuple

from app.schemas import ChatMessage


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    def stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:
        """Yield generated text pieces (implemented as an async generator)."""
        ...


class ASRBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def transcribe(self, wav_bytes: bytes) -> Tuple[str, dict]:
        """Return (transcript, info)."""
        ...


class TTSBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def synthesize(
        self, text: str, *, voice: Optional[str], speed: Optional[float]
    ) -> Tuple[bytes, int]:
        """Return (wav_bytes, sample_rate)."""
        ...
