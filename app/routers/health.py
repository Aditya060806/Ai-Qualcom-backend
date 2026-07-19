"""/health -- liveness + capability report for the dashboard heartbeat."""
from __future__ import annotations

import platform
import sys
import time

from fastapi import APIRouter

from app.schemas import BackendInfo, HealthResponse
from app.state import state
from config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    if not state.ready:
        status = "starting"
    elif all(state.backend_loaded.values()):
        status = "ok"
    else:
        status = "degraded"
    return HealthResponse(
        status=status,
        mode=settings.mode,
        uptime_seconds=round(time.time() - state.start_time, 1),
        warmup_complete=state.warmup_complete,
        backends=BackendInfo(
            llm=settings.resolved_llm_backend,
            asr=settings.resolved_asr_backend,
            tts=settings.resolved_tts_backend,
        ),
        backends_loaded=state.backend_loaded,
        backend_errors=state.backend_errors,
        npu=state.npu_info,
        versions={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    )
