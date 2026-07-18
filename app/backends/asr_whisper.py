"""Whisper ASR backend.

Default implementation uses faster-whisper (CTranslate2), which is easy to
stand up and gives a working /asr on day one. On the X Elite the NPU path is the
QNN / simple-whisper-transcription build the plan references -- when that bundle
is on the device, swap the import and the model construction below; the
``transcribe`` contract (bytes in, text out) stays the same.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Tuple

from app.backends.base import ASRBackend
from app.logging_conf import get_logger
from config import get_settings

log = get_logger("asr.whisper")


class WhisperASR(ASRBackend):
    name = "whisper"

    def __init__(self) -> None:
        self._model = None

    async def load(self) -> None:
        settings = get_settings()
        from faster_whisper import WhisperModel  # lazy

        def _load():
            model_id = settings.whisper_model_path or settings.whisper_model_size
            device = "cuda" if settings.whisper_device == "cuda" else "cpu"
            return WhisperModel(
                model_id, device=device, compute_type=settings.whisper_compute_type
            )

        log.info("Loading Whisper model (%s) ...",
                 settings.whisper_model_path or settings.whisper_model_size)
        self._model = await asyncio.to_thread(_load)
        log.info("Whisper model loaded.")

    async def transcribe(self, wav_bytes: bytes) -> Tuple[str, dict]:
        if self._model is None:
            raise RuntimeError("WhisperASR.load() was not called or failed")

        def _run():
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                tmp.write(wav_bytes)
                tmp.close()
                segments, info = self._model.transcribe(tmp.name, beam_size=1)
                text = "".join(seg.text for seg in segments).strip()
                return text, {
                    "backend": "faster-whisper",
                    "language": getattr(info, "language", None),
                    "duration": getattr(info, "duration", None),
                }
            finally:
                os.unlink(tmp.name)

        return await asyncio.to_thread(_run)
