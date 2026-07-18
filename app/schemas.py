"""Request/response models for the Brain Node HTTP API.

The chat contract is intentionally OpenAI-shaped so the Command PC hub (and any
OpenAI client, including AnythingLLM tooling) can talk to it unchanged.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # When False (default) the server prepends its own system prompt carrying the
    # [EMOTION:xxx] contract, unless the caller already supplied a system message.
    raw: bool = False


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    format: Literal["wav"] = "wav"


class BackendInfo(BaseModel):
    llm: str
    asr: str
    tts: str


class HealthResponse(BaseModel):
    status: str  # "starting" | "ok" (all backends loaded) | "degraded" (some failed)
    mode: str
    uptime_seconds: float
    warmup_complete: bool
    backends: BackendInfo
    backends_loaded: dict
    backend_errors: dict
    npu: dict
    versions: dict
    node: str = "brain-node"
