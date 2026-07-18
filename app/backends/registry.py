"""Backend factory. Builds the three backends from resolved settings.

Imports are local to each branch so selecting the mock backends never imports
onnxruntime / whisper / kokoro.
"""
from __future__ import annotations

from app.backends.base import ASRBackend, LLMBackend, TTSBackend
from config import Settings


def build_llm(settings: Settings) -> LLMBackend:
    backend = settings.resolved_llm_backend
    if backend == "mock":
        from app.backends.llm_mock import MockLLM
        return MockLLM()
    if backend == "genai":
        from app.backends.llm_genai import GenAILLM
        return GenAILLM()
    if backend == "anythingllm":
        from app.backends.llm_anythingllm import AnythingLLMProxy
        return AnythingLLMProxy()
    if backend == "geniex":
        from app.backends.llm_geniex import GenieXProxy
        return GenieXProxy()
    raise ValueError(f"Unknown LLM backend: {backend!r}")


def build_asr(settings: Settings) -> ASRBackend:
    backend = settings.resolved_asr_backend
    if backend == "mock":
        from app.backends.asr_mock import MockASR
        return MockASR()
    if backend in ("onnx", "onnx-asr"):
        from app.backends.asr_onnx import OnnxASR
        return OnnxASR()
    if backend == "whisper":
        from app.backends.asr_whisper import WhisperASR
        return WhisperASR()
    raise ValueError(f"Unknown ASR backend: {backend!r}")


def build_tts(settings: Settings) -> TTSBackend:
    backend = settings.resolved_tts_backend
    if backend == "mock":
        from app.backends.tts_mock import MockTTS
        return MockTTS()
    if backend == "kokoro":
        from app.backends.tts_kokoro import KokoroTTS
        return KokoroTTS()
    raise ValueError(f"Unknown TTS backend: {backend!r}")
