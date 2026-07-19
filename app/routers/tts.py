"""/tts -- text -> WAV audio bytes."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app.schemas import TTSRequest
from app.state import state

router = APIRouter()


@router.post("/tts")
async def tts(req: TTSRequest):
    if state.tts is None:
        return JSONResponse({"error": "tts backend not ready"}, status_code=503)
    wav_bytes, sample_rate = await state.tts.synthesize(
        req.text, voice=req.voice, speed=req.speed
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "Content-Disposition": 'inline; filename="tts.wav"',
        },
    )
