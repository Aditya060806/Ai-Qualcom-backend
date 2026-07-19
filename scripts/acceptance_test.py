"""Brain Node standalone acceptance test (the plan's node-1 gate).

Runs the three checks against a running server:
  1. /health responds
  2. a WAV round-trips /tts -> /asr (validates the audio plumbing)
  3. /v1/chat streams tokens back, and client-side TTFT is measured

    python scripts/acceptance_test.py --base http://127.0.0.1:8080

Exit code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def check_health(base: str) -> dict:
    resp = httpx.get(f"{base}/health", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print("[health]", json.dumps(data, indent=2))
    return data


def check_tts(base: str, text: str) -> bytes:
    resp = httpx.post(f"{base}/tts", json={"text": text}, timeout=60)
    resp.raise_for_status()
    wav = resp.content
    print(f"[tts] {len(wav)} bytes, sample_rate={resp.headers.get('X-Sample-Rate')}")
    assert wav[:4] == b"RIFF", "TTS did not return a RIFF/WAV payload"
    return wav


def check_asr(base: str, wav: bytes) -> dict:
    files = {"file": ("test.wav", wav, "audio/wav")}
    resp = httpx.post(f"{base}/asr", files=files, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print("[asr]", json.dumps(data, indent=2))
    assert "text" in data, "ASR response missing 'text'"
    return data


def check_chat(base: str, prompt: str):
    payload = {"messages": [{"role": "user", "content": prompt}], "stream": True}
    started = time.perf_counter()
    ttft_ms = None
    pieces = []
    metrics = None
    with httpx.stream("POST", f"{base}/v1/chat", json=payload, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("object") == "chat.completion.chunk":
                delta = obj["choices"][0]["delta"].get("content")
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    pieces.append(delta)
            elif obj.get("object") == "brutus.metrics":
                metrics = obj
    reply = "".join(pieces)
    print(f"[chat] reply: {reply!r}")
    if ttft_ms is not None:
        print(f"[chat] client TTFT: {ttft_ms:.1f} ms")
    else:
        print("[chat] no tokens received!")
    print(f"[chat] server metrics: {json.dumps(metrics)}")
    assert reply, "chat returned no text"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt", default="Say hello to the hackathon judges.")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"== Brutus Brain Node acceptance test @ {base} ==")
    try:
        health = check_health(base)
        if health.get("status") != "ok":
            print("[warn] health status is not 'ok' yet (still warming up?)")
        wav = check_tts(base, "Hello from the brain node acceptance test.")
        check_asr(base, wav)
        check_chat(base, args.prompt)
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        print("== FAIL ==")
        return 1
    print("== PASS ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
