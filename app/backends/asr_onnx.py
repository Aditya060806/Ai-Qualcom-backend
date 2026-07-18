"""ONNX Runtime ASR backend (onnx-asr).

This is the default real ASR on Windows ARM64: it runs Whisper (and other) ONNX
models directly on onnxruntime -- no torch, no CTranslate2 (faster-whisper's
engine has no win-arm64 wheel). The model (default ``whisper-base``) is
downloaded from Hugging Face on first load. onnx-asr also accepts a ``providers``
list, so the model can later be pointed at the QNN/NPU EP.

Audio is decoded from WAV bytes to a float32 mono array with the standard library
(so we don't add a soundfile dependency just to read audio); onnx-asr resamples
to the model's rate internally.
"""
from __future__ import annotations

import asyncio
import io
import wave
from typing import Tuple

from app.backends.base import ASRBackend
from app.logging_conf import get_logger
from config import get_settings

log = get_logger("asr.onnx")


def _wav_to_float32_mono(wav_bytes: bytes):
    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if sample_width == 1:  # 8-bit PCM is unsigned
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


class OnnxASR(ASRBackend):
    name = "onnx-asr"

    def __init__(self) -> None:
        self._model = None

    async def load(self) -> None:
        settings = get_settings()
        import onnx_asr

        def _load():
            # Default providers (CPU EP). Pass providers=[...] here to target QNN
            # once a QNN-compatible ASR model is in use.
            return onnx_asr.load_model(settings.asr_model)

        log.info("Loading onnx-asr model %r ...", settings.asr_model)
        self._model = await asyncio.to_thread(_load)
        log.info("onnx-asr model loaded.")

    async def transcribe(self, wav_bytes: bytes) -> Tuple[str, dict]:
        if self._model is None:
            raise RuntimeError("OnnxASR.load() was not called or failed")
        settings = get_settings()

        def _run():
            waveform, sample_rate = _wav_to_float32_mono(wav_bytes)
            text = self._model.recognize(waveform, sample_rate=sample_rate)
            if isinstance(text, list):
                text = text[0] if text else ""
            return text.strip(), {
                "backend": "onnx-asr",
                "model": settings.asr_model,
                "sample_rate": sample_rate,
            }

        return await asyncio.to_thread(_run)
