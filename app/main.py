"""Brutus Brain Node -- FastAPI application (control path).

Boot sequence (lifespan):
  1. verify the execution providers / NPU and log the proof
  2. build + load the three backends exactly once
  3. fire a warm-up inference so there is no cold start during the demo
  4. flip ready=True

Run with a single worker so models load once and stay resident:
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.backends.registry import build_asr, build_llm, build_tts
from app.logging_conf import configure_logging, get_logger
from app.npu import verify_npu
from app.routers import asr as asr_router
from app.routers import chat as chat_router
from app.routers import health as health_router
from app.routers import tts as tts_router
from app.state import state
from app.warmup import run_warmup
from config import get_settings

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("=== Brutus Brain Node starting (mode=%s) ===", settings.mode)

    # 1) NPU / execution-provider verification -- logged for the boot record.
    state.npu_info = verify_npu()
    log.info(
        "NPU verification: ort=%s providers=%s qnn_available=%s",
        state.npu_info.get("onnxruntime_version"),
        state.npu_info.get("providers"),
        state.npu_info.get("qnn_available"),
    )
    if state.npu_info.get("ep_devices"):
        log.info("EP devices: %s", state.npu_info["ep_devices"])
    if settings.mode == "real" and not state.npu_info.get("qnn_available"):
        log.warning(
            "QNN execution provider NOT detected -- the LLM will not run on the NPU. "
            "Check the onnxruntime-qnn install and the model's genai_config.json."
        )

    # 2) Build + load each backend independently. A failure (typically the LLM
    #    before its bundle is built) is logged and that backend is left None --
    #    the node still boots and serves the others. Never crash at boot.
    async def _init(kind: str, builder):
        name = "?"
        try:
            backend = builder(settings)
            name = backend.name
            await backend.load()
        except Exception as e:
            log.error("Backend '%s' (%s) unavailable: %s", kind, name, e)
            state.backend_errors[kind] = f"{type(e).__name__}: {e}"
            return None
        state.backend_loaded[kind] = True
        log.info("Backend '%s' ready: %s", kind, name)
        return backend

    state.llm = await _init("llm", build_llm)
    state.asr = await _init("asr", build_asr)
    state.tts = await _init("tts", build_tts)
    loaded = [k for k, v in state.backend_loaded.items() if v]
    log.info("Backends loaded: %s", ", ".join(loaded) if loaded else "none")

    # 3) Warm-up.
    if settings.warmup_enabled:
        await run_warmup(settings.warmup_prompt)
    else:
        state.warmup_complete = True

    state.ready = True
    log.info("=== Brain Node READY on %s:%d ===", settings.host, settings.port)
    yield
    log.info("Brain Node shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Brutus Brain Node", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    if settings.api_key:
        @app.middleware("http")
        async def require_bearer(request: Request, call_next):
            # /health stays open so dashboards can poll without a token.
            if request.url.path == "/health":
                return await call_next(request)
            if request.headers.get("Authorization", "") != f"Bearer {settings.api_key}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    app.include_router(health_router.router)
    app.include_router(chat_router.router)
    app.include_router(asr_router.router)
    app.include_router(tts_router.router)

    @app.get("/")
    async def root():
        return {
            "node": "brain-node",
            "status": "ok" if state.ready else "starting",
            "endpoints": ["/health", "/v1/chat", "/asr", "/tts"],
        }

    return app


app = create_app()
