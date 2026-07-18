"""Process-wide runtime state.

Backends are loaded exactly once at boot (see ``app.main`` lifespan) and held
resident here for the life of the process. Run the server with a single worker
so this state -- and the loaded models -- are not duplicated across processes.
"""
from __future__ import annotations

import time
from typing import Optional

from app.backends.base import ASRBackend, LLMBackend, TTSBackend


class AppState:
    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.llm: Optional[LLMBackend] = None
        self.asr: Optional[ASRBackend] = None
        self.tts: Optional[TTSBackend] = None
        self.warmup_complete: bool = False
        self.ready: bool = False
        self.npu_info: dict = {}
        # Per-backend readiness. A backend that fails to load (e.g. the LLM
        # before its bundle exists) is set to None above and recorded here, so
        # the node keeps serving the others instead of crashing at boot.
        self.backend_loaded: dict = {"llm": False, "asr": False, "tts": False}
        self.backend_errors: dict = {}


state = AppState()
