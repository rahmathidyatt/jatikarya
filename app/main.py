"""Server web JATIKARYA.

Jalankan:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Variabel lingkungan:
  JATIKARYA_REGISTRY  path ke registry.npz (hasil scripts/build_registry.py)
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import Verifier

STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="JATIKARYA – Verifikasi Asal Gambar")
_verifier: Verifier | None = None
_load_error: str | None = None


@app.on_event("startup")
def _load() -> None:
    global _verifier, _load_error
    try:
        _verifier = Verifier(os.environ.get("JATIKARYA_REGISTRY", "registry.npz"))
    except Exception as e:  # tampilkan di /api/health, jangan matikan server
        _load_error = f"{type(e).__name__}: {e}"


@app.get("/api/health")
def health() -> dict:
    if _verifier is None:
        return {"ready": False, "error": _load_error or "model belum dimuat"}
    return {
        "ready": True,
        "candidates": len(_verifier.registry.numbers),
        "device": _verifier.dino.device,
        "threshold": _verifier.calibrator.threshold,
    }


@app.post("/api/verify")
async def verify(file: UploadFile = File(...)) -> dict:
    if _verifier is None:
        raise HTTPException(503, _load_error or "Model belum siap")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Berkas kosong")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Berkas lebih dari 25 MB")
    try:
        return _verifier.verify(data, file.filename or "")
    except Exception as e:
        raise HTTPException(422, f"Gambar tidak bisa diproses: {e}") from e


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
