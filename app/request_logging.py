"""Comprehensive per-request access logging as a TRANSPARENT ASGI wrapper.

This logs EVERY HTTP request to the Brain Node without touching app.main, the
routers, or any endpoint. It imports the existing FastAPI app unmodified and
wraps it in a thin ASGI layer that records, per request:

  - client ip:port, method, path, query, HTTP version
  - request headers (Authorization/Cookie/API-key values redacted)
  - request body (for text/JSON/form content, truncated) + exact byte size
  - response status, response headers, response body (text/SSE truncated) + size
  - duration in ms

Two files are written (both rotating):
  - logs/requests.log            one concise line per request (quick scan)
  - logs/requests.detailed.jsonl one JSON object per request (everything)

Lifespan / websocket scopes are forwarded untouched, so boot/warm-up and all
endpoint behaviour (including streaming /v1/chat and multipart /asr uploads) are
IDENTICAL to app.main:app -- this layer only observes.

Run the server against this module to enable logging:

    python -m uvicorn app.request_logging:app --host 0.0.0.0 --port 8080 --workers 1

Config (env):
    BRUTUS_ACCESS_LOG            concise log path   (default logs/requests.log)
    BRUTUS_ACCESS_LOG_DETAIL     detailed jsonl path(default logs/requests.detailed.jsonl)
    BRUTUS_ACCESS_LOG_BODIES     "1"/"0" capture bodies (default "1")
    BRUTUS_ACCESS_REQ_BODY_CAP   max chars of request body logged  (default 4096)
    BRUTUS_ACCESS_RESP_BODY_CAP  max chars of response body logged (default 8192)
"""
from __future__ import annotations

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

from app.main import app as _base_app  # existing FastAPI app, imported unmodified

# --- configuration -----------------------------------------------------------
_CONCISE_PATH = os.environ.get("BRUTUS_ACCESS_LOG", os.path.join("logs", "requests.log"))
_DETAIL_PATH = os.environ.get("BRUTUS_ACCESS_LOG_DETAIL", os.path.join("logs", "requests.detailed.jsonl"))
_LOG_BODIES = os.environ.get("BRUTUS_ACCESS_LOG_BODIES", "1").lower() not in ("0", "false", "no", "")
_REQ_CAP = int(os.environ.get("BRUTUS_ACCESS_REQ_BODY_CAP", "4096"))
_RESP_CAP = int(os.environ.get("BRUTUS_ACCESS_RESP_BODY_CAP", "8192"))

_SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key", "proxy-authorization"}
_TEXTUAL = ("json", "text", "xml", "x-www-form-urlencoded", "event-stream", "javascript")


def _setup_logger(name: str, path: str, formatter: logging.Formatter, max_bytes: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep out of the app's stdout logger
    if not any(getattr(h, "_brutus_access", False) for h in logger.handlers):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=5, encoding="utf-8")
        handler._brutus_access = True  # marker so re-import doesn't duplicate handlers
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


concise_logger = _setup_logger(
    "brutus.access", _CONCISE_PATH,
    logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"), 5_000_000,
)
detail_logger = _setup_logger(
    "brutus.access.detail", _DETAIL_PATH, logging.Formatter("%(message)s"), 20_000_000,
)


def _redacted_headers(raw_headers) -> dict:
    out = {}
    for key_b, val_b in raw_headers or []:
        key = key_b.decode("latin-1").lower()
        val = val_b.decode("latin-1")
        out[key] = "***redacted***" if key in _SENSITIVE_HEADERS else val
    return out


def _maybe_text_body(body: bytes, content_type: str, cap: int):
    """Return a truncated text form of the body for text/JSON/form/SSE content,
    or None for binary (we still log the byte size separately)."""
    if not _LOG_BODIES or not body:
        return None
    ct = (content_type or "").lower()
    if not any(token in ct for token in _TEXTUAL):
        return None  # binary (audio/*, multipart, octet-stream, ...) -> size only
    text = body[:cap].decode("utf-8", errors="replace")
    if len(body) > cap:
        text += f"...(+{len(body) - cap} more bytes)"
    return text


class RequestLoggingMiddleware:
    """Pure ASGI middleware. Buffers the request body then replays it to the app,
    captures the response as it streams out, and logs both. Non-HTTP scopes
    (lifespan/websocket) pass straight through."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()

        # --- buffer the full request body, then hand the app a replay ---------
        request_chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                request_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break
        request_body = b"".join(request_chunks)

        replay_state = {"sent": False}

        async def receive_replay():
            if not replay_state["sent"]:
                replay_state["sent"] = True
                return {"type": "http.request", "body": request_body, "more_body": False}
            # After the (buffered) body, defer to the real transport so streaming
            # responses' disconnect-listener blocks for a REAL disconnect instead
            # of aborting early on a fake one.
            return await receive()

        # --- capture the response as it is sent -------------------------------
        resp = {"status": 0, "headers": []}
        resp_size = {"n": 0}
        resp_cap = bytearray()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                resp["status"] = message["status"]
                resp["headers"] = message.get("headers", [])
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                resp_size["n"] += len(chunk)
                if _LOG_BODIES and len(resp_cap) < _RESP_CAP:
                    resp_cap.extend(chunk[: _RESP_CAP - len(resp_cap)])
            await send(message)

        error_repr = None
        try:
            await self.app(scope, receive_replay, send_wrapper)
        except Exception as e:  # log then re-raise so behaviour is unchanged
            error_repr = repr(e)
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            client = scope.get("client")
            client_str = f"{client[0]}:{client[1]}" if client else "-"
            method = scope.get("method", "-")
            path = scope.get("path", "-")
            query = scope.get("query_string", b"").decode("latin-1")
            full_path = f"{path}?{query}" if query else path
            req_headers = _redacted_headers(scope.get("headers", []))
            resp_headers = _redacted_headers(resp["headers"])
            req_ct = req_headers.get("content-type", "")
            resp_ct = resp_headers.get("content-type", "")

            # concise one-liner
            concise_logger.info(
                "%s %s %s %d %dB %.1fms",
                client_str, method, full_path, resp["status"], resp_size["n"], duration_ms,
            )

            # full detail as one JSON line
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "client": client_str,
                "method": method,
                "path": path,
                "query": query,
                "http_version": scope.get("http_version"),
                "status": resp["status"],
                "duration_ms": duration_ms,
                "request": {
                    "content_type": req_ct,
                    "body_bytes": len(request_body),
                    "body": _maybe_text_body(request_body, req_ct, _REQ_CAP),
                    "headers": req_headers,
                },
                "response": {
                    "content_type": resp_ct,
                    "body_bytes": resp_size["n"],
                    "body": _maybe_text_body(bytes(resp_cap), resp_ct, _RESP_CAP),
                    "headers": resp_headers,
                },
            }
            if error_repr:
                record["error"] = error_repr
            try:
                detail_logger.info(json.dumps(record, ensure_ascii=False))
            except Exception:
                pass  # logging must never break a request


# The ASGI app uvicorn should serve to get request logging.
app = RequestLoggingMiddleware(_base_app)
