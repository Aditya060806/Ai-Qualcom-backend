"""GenieX proxy LLM backend (Qualcomm on-device NPU LLM).

GenieX runs the AI Hub Llama 3.2 3B (SSD) model on the Hexagon NPU and exposes an
OpenAI-compatible local server (default http://127.0.0.1:18181). This backend
just streams from that server, which keeps GenieX's QAIRT/NPU runtime isolated in
its own process from the onnxruntime-qnn stack this server already loads -- no two
QNN runtimes fighting over the NPU inside one process.

Order of operations:
  1. geniex serve            # starts the OpenAI server on :18181 (NPU)
  2. BRUTUS_LLM_BACKEND=geniex  + run the Brain Node

If the GenieX server is not reachable at boot, load() raises and the node's
graceful degradation marks the LLM unavailable (/health degraded, /chat -> 503)
while ASR and TTS keep working.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, List

import httpx

from app.backends.base import LLMBackend
from app.logging_conf import get_logger
from app.schemas import ChatMessage
from config import get_settings

log = get_logger("llm.geniex")


class GenieXProxy(LLMBackend):
    name = "geniex"

    def _headers(self) -> dict:
        settings = get_settings()
        headers = {"Content-Type": "application/json"}
        if settings.geniex_api_key:
            headers["Authorization"] = f"Bearer {settings.geniex_api_key}"
        return headers

    async def load(self) -> None:
        settings = get_settings()
        base = settings.geniex_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{base}/models", headers=self._headers())
                resp.raise_for_status()
            log.info("GenieX server reachable at %s (model=%s)", base, settings.geniex_model)
        except Exception as e:
            raise RuntimeError(
                f"GenieX server not reachable at {base} ({e}). "
                "Start it first with `geniex serve`."
            ) from e

    async def stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:
        settings = get_settings()
        url = settings.geniex_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": settings.geniex_model,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": [m.model_dump() for m in messages],
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
