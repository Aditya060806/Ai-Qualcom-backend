"""Mock ASR. Returns a canned transcript but parses the WAV header so the
reported duration reflects the real uploaded audio -- enough to validate the
/asr plumbing end-to-end.
"""
from __future__ import annotations

import io
import wave
from typing import Tuple

from app.backends.base import ASRBackend


class MockASR(ASRBackend):
    name = "mock"

    async def load(self) -> None:
        return None

    async def transcribe(self, wav_bytes: bytes) -> Tuple[str, dict]:
        duration = None
        sample_rate = None
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                frames = w.getnframes()
                sample_rate = w.getframerate()
                if sample_rate:
                    duration = round(frames / float(sample_rate), 3)
        except Exception:
            pass
        text = "This is a mock transcript from the Brutus brain node."
        return text, {
            "backend": "mock",
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "bytes": len(wav_bytes),
        }
