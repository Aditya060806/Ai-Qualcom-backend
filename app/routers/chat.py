"""/v1/chat -- OpenAI-shaped chat, streaming (SSE) or single-shot.

Streaming responses are emitted as ``chat.completion.chunk`` events, then a
custom ``brutus.metrics`` event (TTFT, tokens/sec, parsed emotion), then the
terminal ``[DONE]``. Non-OpenAI clients can ignore the metrics event.
"""
from __future__ import annotations

import json
import re
import time
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.metrics import GenerationMetrics
from app.schemas import ChatMessage, ChatRequest
from app.state import state
from config import get_settings

router = APIRouter()

_EMOTION_RE = re.compile(r"\[EMOTION:([a-zA-Z]+)\]")


def _extract_emotion(text: str) -> Optional[str]:
    match = _EMOTION_RE.search(text)
    return match.group(1).lower() if match else None


def _prepare_messages(req: ChatRequest) -> List[ChatMessage]:
    settings = get_settings()
    messages = list(req.messages)
    suffix = f" {settings.llm_system_suffix}" if settings.llm_system_suffix else ""
    has_system = any(m.role == "system" for m in messages)
    if not req.raw and not has_system:
        messages = [ChatMessage(role="system", content=settings.system_prompt + suffix)] + messages
    elif suffix:
        for i, m in enumerate(messages):
            if m.role == "system":
                messages[i] = ChatMessage(role="system", content=m.content + suffix)
                break
    return messages


@router.post("/v1/chat")
@router.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    settings = get_settings()
    if state.llm is None:
        return JSONResponse({"error": "llm backend not ready"}, status_code=503)

    messages = _prepare_messages(req)
    max_tokens = req.max_tokens or settings.llm_max_tokens
    temperature = req.temperature if req.temperature is not None else settings.llm_temperature
    model = req.model or "brutus-edge"

    if req.stream:
        async def event_stream():
            metrics = GenerationMetrics()
            created = int(time.time())
            collected: List[str] = []
            async for piece in state.llm.stream(
                messages, max_tokens=max_tokens, temperature=temperature, top_p=settings.llm_top_p
            ):
                metrics.on_token()
                collected.append(piece)
                chunk = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"

            summary = metrics.summary()
            summary["emotion"] = _extract_emotion("".join(collected))
            yield f"data: {json.dumps({'object': 'brutus.metrics', **summary})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: collect the full reply.
    metrics = GenerationMetrics()
    collected = []
    async for piece in state.llm.stream(
        messages, max_tokens=max_tokens, temperature=temperature, top_p=settings.llm_top_p
    ):
        metrics.on_token()
        collected.append(piece)
    text = "".join(collected)
    created = int(time.time())
    return JSONResponse(
        {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": metrics.tokens},
            "brutus": {**metrics.summary(), "emotion": _extract_emotion(text)},
        }
    )
