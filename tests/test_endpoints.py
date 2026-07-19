"""Endpoint tests against the mock backends (no NPU/models required).

TestClient used as a context manager runs the FastAPI lifespan, so backends are
built, loaded and warmed exactly as in production.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_mock_backends():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "mock"
        assert data["backends"] == {"llm": "mock", "asr": "mock", "tts": "mock"}
        assert data["warmup_complete"] is True


def test_tts_returns_wav_and_asr_accepts_it():
    with TestClient(app) as client:
        tts_resp = client.post("/tts", json={"text": "hello world"})
        assert tts_resp.status_code == 200
        assert tts_resp.content[:4] == b"RIFF"

        files = {"file": ("t.wav", tts_resp.content, "audio/wav")}
        asr_resp = client.post("/asr", files=files)
        assert asr_resp.status_code == 200
        body = asr_resp.json()
        assert "text" in body
        assert body["info"]["sample_rate"] == 22050


def test_chat_streams_chunks_and_done():
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "chat.completion.chunk" in body
        assert "brutus.metrics" in body
        assert "[DONE]" in body
        assert "[EMOTION:" in body


def test_chat_non_streaming_returns_message():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content
        assert data["brutus"]["emotion"] == "happy"
