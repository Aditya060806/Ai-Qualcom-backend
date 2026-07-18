"""Kokoro TTS backend (offline, ONNX).

Uses kokoro-onnx, which runs fully offline and cross-platform. Produces float32
samples that we pack into a 16-bit PCM WAV. Piper is the low-end fallback (see
requirements-npu.txt) if Kokoro is too heavy on a given device.
"""
from __future__ import annotations

import asyncio
import io
import wave
from typing import Optional, Tuple

from app.backends.base import TTSBackend
from app.logging_conf import get_logger
from config import get_settings

log = get_logger("tts.kokoro")


class KokoroTTS(TTSBackend):
    name = "kokoro"

    def __init__(self) -> None:
        self._kokoro = None

    async def load(self) -> None:
        settings = get_settings()
        from kokoro_onnx import Kokoro  # lazy

        # Point Kokoro's phonemizer at the bundled espeak-ng (no system install).
        espeak_config = None
        try:
            import espeakng_loader
            from kokoro_onnx import EspeakConfig

            espeak_config = EspeakConfig(
                lib_path=espeakng_loader.get_library_path(),
                data_path=espeakng_loader.get_data_path(),
            )
        except Exception as e:
            log.info("espeakng-loader not configured (%s); relying on Kokoro defaults", e)

        def _load():
            return Kokoro(
                settings.kokoro_model_path,
                settings.kokoro_voices_path,
                espeak_config=espeak_config,
            )

        log.info("Loading Kokoro model from %s ...", settings.kokoro_model_path)
        self._kokoro = await asyncio.to_thread(_load)
        log.info("Kokoro model loaded.")

    async def synthesize(
        self, text: str, *, voice: Optional[str] = None, speed: Optional[float] = None
    ) -> Tuple[bytes, int]:
        if self._kokoro is None:
            raise RuntimeError("KokoroTTS.load() was not called or failed")
        settings = get_settings()

        def _run():
            import numpy as np

            samples, sample_rate = self._kokoro.create(
                text,
                voice=voice or settings.kokoro_voice,
                speed=speed or settings.kokoro_speed,
                lang=settings.kokoro_lang,
            )
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm)
            return buffer.getvalue(), int(sample_rate)

        return await asyncio.to_thread(_run)
