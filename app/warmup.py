"""Boot-time warm-up.

The first inference on the NPU (and on most model runtimes) is dramatically
slower than steady state. We fire a throwaway generation and a throwaway TTS
synth at startup so the first *real* request a judge triggers is already warm.
"""
from __future__ import annotations

import time

from app.logging_conf import get_logger
from app.schemas import ChatMessage
from app.state import state

log = get_logger("warmup")


async def run_warmup(prompt: str) -> None:
    if state.llm is not None:
        started = time.perf_counter()
        try:
            chunks = 0
            async for _ in state.llm.stream(
                [ChatMessage(role="user", content=prompt)],
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
            ):
                chunks += 1
            log.info(
                "LLM warm-up complete (%d chunks, %.0f ms)",
                chunks,
                (time.perf_counter() - started) * 1000,
            )
        except Exception as e:  # never let warm-up crash boot
            log.warning("LLM warm-up failed: %s", e)

    if state.tts is not None:
        started = time.perf_counter()
        try:
            await state.tts.synthesize("warm up", voice=None, speed=None)
            log.info("TTS warm-up complete (%.0f ms)", (time.perf_counter() - started) * 1000)
        except Exception as e:
            log.warning("TTS warm-up failed: %s", e)

    state.warmup_complete = True
