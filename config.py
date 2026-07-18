"""Central configuration for the Brutus Brain Node.

All settings are read from environment variables prefixed with ``BRUTUS_`` (or a
local ``.env`` file). Two runtime modes exist:

* ``mock``  - no heavy dependencies; deterministic responses. Lets every other
  node (Command PC hub, phone, Sense Station) develop against a real HTTP
  contract on any machine, and lets us verify plumbing without the NPU.
* ``real``  - loads the on-device backends (onnxruntime-genai on the QNN/NPU
  execution provider, Whisper, Kokoro) on the Snapdragon X Elite.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BRUTUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime mode -------------------------------------------------------
    mode: str = "mock"  # "mock" | "real"

    # --- Server -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    # Optional bearer token. Empty string = open on the LAN (typical for the
    # closed hackathon network). Set BRUTUS_API_KEY to require
    # "Authorization: Bearer <token>" on every endpoint except /health.
    api_key: str = ""

    # Comma-separated CORS origins for the dashboard. "*" allows any origin.
    cors_allow_origins: str = "*"

    # --- Backend selection (override the mode defaults if set) --------------
    llm_backend: Optional[str] = None   # mock | genai | anythingllm
    asr_backend: Optional[str] = None   # mock | whisper
    tts_backend: Optional[str] = None   # mock | kokoro

    # --- LLM ----------------------------------------------------------------
    llm_model_path: str = "models/llm/llama-3.2-3b-genai"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    # If set (e.g. "QNNExecutionProvider"), the genai backend clears the model's
    # configured providers and forces this one -- use it to pin the LLM to the NPU.
    # If empty, the model's own genai_config.json decides.
    llm_force_provider: str = ""
    # Appended to the system prompt (any backend). Defaults to "/no_think" so the
    # locked-in Qwen3-4B skips its verbose <think> reasoning block (clean, fast
    # replies). Mock ignores it. If you switch to a NON-Qwen model (e.g. a Llama
    # genai bundle), set BRUTUS_LLM_SYSTEM_SUFFIX="" to disable it.
    llm_system_suffix: str = "/no_think"
    system_prompt: str = (
        "You are Brutus, an offline voice assistant running entirely on edge "
        "hardware. Keep replies short and easy to speak aloud. Begin every reply "
        "with exactly one emotion tag chosen from [EMOTION:neutral], "
        "[EMOTION:happy], [EMOTION:sad], [EMOTION:thinking], [EMOTION:surprised], "
        "[EMOTION:angry], then give your spoken answer."
    )

    # AnythingLLM proxy backend (the "fast path"). Point at its OpenAI-compatible
    # API. Used only when the resolved LLM backend is "anythingllm".
    anythingllm_base_url: str = "http://localhost:3001/api/v1"
    anythingllm_api_key: str = ""
    anythingllm_workspace: str = "brutus"

    # GenieX proxy backend (Qualcomm on-device NPU LLM via its OpenAI-compatible
    # local server). Used when the resolved LLM backend is "geniex". geniex_model
    # must match the id the server advertises at /v1/models -- WITHOUT any
    # ":precision" suffix (e.g. "qualcomm/Qwen3-4B", not "qualcomm/Qwen3-4B:W4A16").
    # qualcomm/Qwen3-4B is an ungated AI Hub NPU (w4a16) bundle -- no export needed.
    # For Qwen3, also set BRUTUS_LLM_SYSTEM_SUFFIX=/no_think for clean, fast replies.
    geniex_base_url: str = "http://127.0.0.1:18181/v1"
    geniex_model: str = "qualcomm/Qwen3-4B"
    geniex_api_key: str = ""

    # --- ASR ----------------------------------------------------------------
    # Default backend "onnx" (onnx-asr) runs Whisper ONNX on onnxruntime, which
    # works on Windows ARM64. onnx-asr model id (downloaded from HF on first load):
    asr_model: str = "whisper-base"
    # faster-whisper settings (only used when BRUTUS_ASR_BACKEND=whisper; that
    # backend needs CTranslate2, which has no win-arm64 wheel -- x86/CUDA only).
    whisper_model_size: str = "base"       # tiny | base | small | medium | ...
    whisper_model_path: str = ""           # optional explicit path/dir override
    whisper_device: str = "cpu"            # cpu | cuda
    whisper_compute_type: str = "int8"

    # --- TTS (Kokoro) -------------------------------------------------------
    kokoro_model_path: str = "models/tts/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "models/tts/voices-v1.0.bin"
    kokoro_voice: str = "af_sarah"
    kokoro_speed: float = 1.0
    kokoro_lang: str = "en-us"

    # --- Warm-up ------------------------------------------------------------
    # A dummy inference fired at boot so a judge never triggers a cold start.
    warmup_enabled: bool = True
    warmup_prompt: str = "Say a short hello."

    # ---------------------------------------------------------------- helpers
    @property
    def resolved_llm_backend(self) -> str:
        if self.llm_backend:
            return self.llm_backend
        # Real mode defaults to GenieX (the locked-in Qwen3-4B NPU LLM). Set
        # BRUTUS_LLM_BACKEND=genai to use an onnxruntime-genai bundle instead.
        return "mock" if self.mode == "mock" else "geniex"

    @property
    def resolved_asr_backend(self) -> str:
        if self.asr_backend:
            return self.asr_backend
        return "mock" if self.mode == "mock" else "onnx"

    @property
    def resolved_tts_backend(self) -> str:
        if self.tts_backend:
            return self.tts_backend
        return "mock" if self.mode == "mock" else "kokoro"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> "Settings":
    return Settings()
