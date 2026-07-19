"""/asr -- multipart WAV upload -> transcript."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.state import state

router = APIRouter()


@router.post("/asr")
async def asr(file: UploadFile = File(...)):
    if state.asr is None:
        return JSONResponse({"error": "asr backend not ready"}, status_code=503)
    wav_bytes = await file.read()
    text, info = await state.asr.transcribe(wav_bytes)
    return JSONResponse({"text": text, "info": info})
