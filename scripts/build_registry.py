"""Bangun registry.npz dari 40 foto sumber ASLI (sebelum diberi watermark).

Contoh:
  python scripts/build_registry.py --photos "/content/drive/MyDrive/.../foto_sumber" --out registry.npz

Cara kerja:
  1. Hitung SHA-256 setiap berkas, ID sumber = "SRC_" + 16 karakter pertama.
  2. Cocokkan dengan 40 ID penelitian (app/research/reference_ids.json).
     Berkas yang tidak ada di daftar dilewati dan dilaporkan.
  3. Hitung embedding DINOv2 dengan prapemrosesan yang sama seperti penelitian.
  4. Bandingkan SHA-256 embedding dengan catatan penelitian (informasi saja).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline import DinoEmbedder, load_research_ids, prepare_image  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, help="folder berisi foto sumber asli")
    ap.add_argument("--out", default="registry.npz")
    ap.add_argument("--allow-missing", action="store_true", help="tetap simpan walau < 40 sumber ditemukan")
    a = ap.parse_args()

    ids = {s["source_sha256"]: s for s in load_research_ids()}
    files = sorted(p for p in Path(a.photos).rglob("*") if p.suffix.lower() in EXT)
    print(f"[Step 1] {len(files)} berkas gambar ditemukan di {a.photos}")

    found: dict[str, tuple[dict, Path, bytes]] = {}
    skipped = []
    for p in files:
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in ids:
            found.setdefault(sha, (ids[sha], p, data))
        else:
            skipped.append(p.name)
    print(f"[Step 2] {len(found)}/40 cocok dengan ID penelitian; {len(skipped)} berkas dilewati")
    missing = [s for sha, s in ids.items() if sha not in found]
    for s in missing:
        print(f"         BELUM ADA: Foto Sumber {s['number']} ({s['source_id']})")
    if missing and not a.allow_missing:
        print("Hentikan: semua 40 foto sumber asli dibutuhkan (atau pakai --allow-missing).")
        return 1

    print("[Step 3] Memuat DINOv2 dan menghitung embedding…")
    dino = DinoEmbedder()
    rows = sorted(found.values(), key=lambda t: t[0]["number"])
    embs, same_hash = [], 0
    for meta, path, data in rows:
        img, _ = prepare_image(data)
        raw = dino.embed(img, normalize=False)
        embs.append(raw / (np.linalg.norm(raw) + 1e-12))
        h_raw = hashlib.sha256(raw.astype(np.float32).tobytes()).hexdigest()
        h_norm = hashlib.sha256(embs[-1].astype(np.float32).tobytes()).hexdigest()
        if meta["embedding_sha256"] in (h_raw, h_norm):
            same_hash += 1
        print(f"         Foto Sumber {meta['number']}  {path.name}")

    print(f"[Step 4] Hash embedding identik dengan penelitian: {same_hash}/{len(rows)}")
    if same_hash < len(rows):
        print("         (Beda hash biasanya karena versi torch/GPU. Urutan kandidat tetap bisa dipakai,"
              " tetapi skor bisa sedikit bergeser dari penelitian.)")

    np.savez(
        a.out,
        numbers=np.array([m["number"] for m, _, _ in rows]),
        source_ids=np.array([m["source_id"] for m, _, _ in rows]),
        source_sha256=np.array([m["source_sha256"] for m, _, _ in rows]),
        embeddings=np.stack(embs).astype(np.float32),
    )
    print(f"[Step 5] Selesai: {a.out} ({len(rows)} foto sumber)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
