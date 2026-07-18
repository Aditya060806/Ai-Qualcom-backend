"""Mock TTS. Produces a real, playable 16-bit PCM WAV (a short 440 Hz tone)
using only the standard library, so /tts returns valid audio with no numpy or
model files present.
"""
from __future__ import annotations

import io
import math
import struct
import wave
from typing import Optional, Tuple

from app.backends.base import TTSBackend


class MockTTS(TTSBackend):
    name = "mock"
    sample_rate = 22050

    async def load(self) -> None:
        return None

    async def synthesize(
        self, text: str, *, voice: Optional[str] = None, speed: Optional[float] = None
    ) -> Tuple[bytes, int]:
        duration = max(0.3, min(3.0, len(text) * 0.03))
        sr = self.sample_rate
        n_frames = int(duration * sr)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            frames = bytearray()
            for i in range(n_frames):
                value = int(32767 * 0.2 * math.sin(2 * math.pi * 440 * (i / sr)))
                frames += struct.pack("<h", value)
            w.writeframes(bytes(frames))
        return buffer.getvalue(), sr
