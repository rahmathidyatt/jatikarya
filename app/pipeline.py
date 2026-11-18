"""Alur verifikasi JATIKARYA: satu gambar masuk, enam tahap, satu keputusan keluar.

Urutan tahap mengikuti penelitian:
  1. Unggah            -> bytes gambar
  2. Persiapan         -> frame 0, EXIF transpose, RGB, SHA-256
  3. TrustMark         -> baca payload 61 bit, petakan ke foto sumber
  4. DINOv2 ViT-S/14   -> embedding 384-d, kemiripan kosinus ke 40 referensi
  5. Calibrated Fusion -> regresi logistik (kemiripan top-1, selisih top-1/top-2)
  6. Keputusan         -> aturan TM_FIRST_THEN_DINOV2 (ACCEPT / ABSTAIN)

torch dan trustmark baru di-import saat model dimuat, sehingga fungsi
kalibrasi dan aturan keputusan bisa diuji tanpa GPU.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RESEARCH_DIR = Path(__file__).resolve().parent / "research"

try:  # foto HEIC dari iPhone
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


# ---------------------------------------------------------------- tahap 2
def prepare_image(data: bytes) -> tuple[Image.Image, dict]:
    """Frame 0 -> EXIF transpose -> RGB, sama seperti EXIF_TRANSPOSE_RGB_FRAME0."""
    img = Image.open(io.BytesIO(data))
    fmt = img.format
    try:
        img.seek(0)
    except EOFError:
        pass
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    info = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": fmt,
        "width": img.width,
        "height": img.height,
        "bytes": len(data),
    }
    return img, info


# ---------------------------------------------------------------- registry
@dataclass
class Registry:
    numbers: list[str]
    source_ids: list[str]
    source_sha256: list[str]
    embeddings: np.ndarray  # (N, 384), sudah dinormalisasi L2

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        z = np.load(path, allow_pickle=False)
        emb = z["embeddings"].astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return cls(
            numbers=[str(x) for x in z["numbers"]],
            source_ids=[str(x) for x in z["source_ids"]],
            source_sha256=[str(x) for x in z["source_sha256"]],
            embeddings=emb,
        )

    def payload_index(self, bits: int = 61) -> dict[str, int]:
        """Payload TrustMark = n bit pertama SHA-256 berkas foto sumber."""
        out = {}
        for i, sha in enumerate(self.source_sha256):
            out[sha256_prefix_bits(sha, bits)] = i
        return out


def sha256_prefix_bits(sha_hex: str, bits: int = 61) -> str:
    return bin(int(sha_hex, 16))[2:].zfill(256)[:bits]


def load_research_ids() -> list[dict]:
    return json.loads((RESEARCH_DIR / "reference_ids.json").read_text())["sources"]


# ---------------------------------------------------------------- tahap 3
class TrustMarkReader:
    def __init__(self, cfg: dict):
        from trustmark import TrustMark  # import di sini supaya tes ringan

        self.cfg = cfg
        enc = getattr(TrustMark.Encoding, cfg["encoding"])
        self.tm = TrustMark(verbose=False, model_type=cfg["model_type"], encoding_type=enc)

    def read(self, img: Image.Image) -> dict:
        t0 = time.perf_counter()
        secret, present, schema = self.tm.decode(img, MODE=self.cfg["mode"])
        if not isinstance(secret, str):
            secret = "".join("1" if int(b) else "0" for b in np.asarray(secret).ravel())
        return {
            "detected": bool(present),
            "payload": secret if present else "",
            "schema": int(schema) if present else -1,
            "seconds": round(time.perf_counter() - t0, 3),
        }


def map_trustmark(raw: dict, registry: Registry, cfg: dict) -> dict:
    """Ubah hasil baca mentah menjadi status yang sama dengan penelitian."""
    res = dict(raw)
    res["source_index"] = None
    if not raw["detected"]:
        res["status"] = "TIDAK_TERDETEKSI"
    elif raw["schema"] != cfg["payload_schema"] or len(raw["payload"]) != cfg["payload_bits"]:
        res["status"] = "SCHEMA_TIDAK_SESUAI"
    else:
        idx = registry.payload_index(cfg["payload_bits"]).get(raw["payload"])
        if idx is None:
            res["status"] = "KODE_TIDAK_DIKENAL"
        else:
            res["status"] = "KODE_COCOK"
            res["source_index"] = idx
    return res


# ---------------------------------------------------------------- tahap 4
class DinoEmbedder:
    def __init__(self, model_name: str = "dinov2_vits14", device: str | None = None):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device).eval()
        # Resize518_CenterCrop518_ImageNet
        self.tf = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def embed(self, img: Image.Image, normalize: bool = True) -> np.ndarray:
        with self.torch.inference_mode():
            x = self.tf(img).unsqueeze(0).to(self.device)
            v = self.model(x).float().cpu().numpy()[0].astype(np.float32)
        return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32) if normalize else v


def rank_candidates(query: np.ndarray, registry: Registry, k: int = 3) -> list[dict]:
    sims = registry.embeddings @ query
    order = np.argsort(-sims)
    return [
        {
            "rank": r + 1,
            "index": int(i),
            "number": registry.numbers[i],
            "source_id": registry.source_ids[i],
            "similarity": float(sims[i]),
        }
        for r, i in enumerate(order[: max(k, 2)])
    ]


# ---------------------------------------------------------------- tahap 5
class Calibrator:
    def __init__(self, cfg: dict):
        self.mean = np.array(cfg["scaler_mean"], dtype=float)
        self.scale = np.array(cfg["scaler_scale"], dtype=float)
        self.coef = np.array(cfg["coefficients"], dtype=float)
        self.intercept = float(cfg["intercept"])
        self.threshold = float(cfg["decision_threshold"])

    def score(self, similarity_top1: float, margin_top1_top2: float) -> float:
        z = (np.array([similarity_top1, margin_top1_top2]) - self.mean) / self.scale
        logit = self.intercept + float(self.coef @ z)
        return 1.0 / (1.0 + math.exp(-logit))


# ---------------------------------------------------------------- tahap 6
def decide(tm_index: int | None, top: list[dict], score: float | None, threshold: float) -> dict:
    """Aturan TM_FIRST_THEN_DINOV2_TOP1 dengan kalibrasi."""
    if not top:
        return {"verdict": "ABSTAIN", "rule": "insufficient_evidence", "source_index": None}
    top1 = top[0]["index"]
    if tm_index is not None:
        if tm_index == top1:
            return {"verdict": "ACCEPT", "rule": "tm_dino_agree", "source_index": tm_index}
        return {"verdict": "ABSTAIN", "rule": "tm_dino_conflict", "source_index": None}
    if score is None or len(top) < 2:
        return {"verdict": "ABSTAIN", "rule": "insufficient_evidence", "source_index": None}
    if score >= threshold:
        return {"verdict": "ACCEPT", "rule": "tm_missing_score_ge_threshold", "source_index": top1}
    return {"verdict": "ABSTAIN", "rule": "tm_missing_score_below_threshold", "source_index": None}


# ---------------------------------------------------------------- semua tahap
class Verifier:
    def __init__(self, registry_path: str | Path, device: str | None = None):
        self.cfg = json.loads((RESEARCH_DIR / "calibration.json").read_text())
        self.registry = Registry.load(registry_path)
        self.calibrator = Calibrator(self.cfg)
        self.trustmark = TrustMarkReader(self.cfg["trustmark"])
        self.dino = DinoEmbedder(self.cfg["dinov2"]["model"], device)

    def verify(self, data: bytes, filename: str = "") -> dict:
        t_all = time.perf_counter()
        img, info = prepare_image(data)
        info["filename"] = filename

        tm = map_trustmark(self.trustmark.read(img), self.registry, self.cfg["trustmark"])

        t0 = time.perf_counter()
        top = rank_candidates(self.dino.embed(img), self.registry, k=3)
        dino_seconds = round(time.perf_counter() - t0, 3)
        sim1 = top[0]["similarity"]
        margin = sim1 - top[1]["similarity"]

        score = self.calibrator.score(sim1, margin)
        decision = decide(tm["source_index"], top, score, self.calibrator.threshold)

        def src(i):
            if i is None:
                return None
            return {"number": self.registry.numbers[i], "source_id": self.registry.source_ids[i]}

        return {
            "stage2_prepare": info,
            "stage3_trustmark": {**tm, "source": src(tm["source_index"])},
            "stage4_dinov2": {"top": top[:3], "similarity_top1": sim1, "margin_top1_top2": margin,
                              "candidates": len(self.registry.numbers), "seconds": dino_seconds},
            "stage5_calibration": {"score": score, "threshold": self.calibrator.threshold,
                                   "used_for_decision": tm["source_index"] is None},
            "stage6_decision": {**decision, "source": src(decision["source_index"])},
            "seconds_total": round(time.perf_counter() - t_all, 3),
        }
