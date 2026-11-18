"""Uji logika tanpa GPU: kalibrasi, pemetaan watermark, aturan keputusan, persiapan gambar.

Jalankan:  python -m unittest discover -s tests -v
"""
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline import (  # noqa: E402
    RESEARCH_DIR, Calibrator, Registry, decide, load_research_ids, map_trustmark,
    prepare_image, rank_candidates, sha256_prefix_bits,
)

CFG = json.loads((RESEARCH_DIR / "calibration.json").read_text())

# Diambil dari final_test_predictions_locked.csv (STEP 11H, 2026-11-10):
# (provider, nomor, edit, s1, margin, reliability_estimate, keputusan)
LOCKED = [
    ("chatgpt", "02", "background", 0.9019817113876344, 0.6106091141700745, 0.997458044063753, "ACCEPT_DINOV2"),
    ("chatgpt", "20", "removal", 0.2905277013778686, 0.0988055765628814, 0.824522451792206, "ACCEPT_DINOV2"),
    ("gemini", "20", "removal", 0.3346292972564697, 0.0522682666778564, 0.8357697820182446, "ACCEPT_DINOV2"),
    ("chatgpt", "16", "replacement", 0.5644165873527527, 0.2260418832302093, 0.9621945913664637, "ACCEPT_DINOV2"),
    ("gemini", "25", "background", 0.6106650829315186, 0.1977603733539581, 0.9670513346403594, "ACCEPT_DINOV2"),
    ("chatgpt", "10", "background", 0.7303286194801331, 0.3355692625045776, 0.9872987003876446, "ACCEPT_DINOV2"),
    ("gemini", "16", "replacement", 0.5822769403457642, 0.2119010090827942, 0.96382621428154, "ACCEPT_AGREEMENT"),
    ("chatgpt", "29", "replacement", 0.9869945645332336, 0.8918040990829468, 0.9992464409926737, "ACCEPT_DINOV2"),
]

# Payload hasil baca TrustMark untuk gemini/removal sumber 01 (trustmark_240_results.csv)
PAYLOAD_SRC01 = "0111010100000100100010101000111011011110101101000011100100101"


def fake_registry(n=40, dim=384, seed=0):
    ids = load_research_ids()[:n]
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, dim)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return Registry([s["number"] for s in ids], [s["source_id"] for s in ids],
                    [s["source_sha256"] for s in ids], e)


class TestCalibration(unittest.TestCase):
    def test_reproduces_locked_research_scores(self):
        cal = Calibrator(CFG)
        for prov, num, edit, s1, m, expected, _ in LOCKED:
            with self.subTest(f"{prov} {num} {edit}"):
                self.assertAlmostEqual(cal.score(s1, m), expected, places=10)

    def test_threshold_decisions_match_research(self):
        cal = Calibrator(CFG)
        for *_, s1, m, _, dec in LOCKED:
            if dec == "ACCEPT_DINOV2":
                self.assertGreaterEqual(cal.score(s1, m), cal.threshold)

    def test_low_evidence_is_below_threshold(self):
        cal = Calibrator(CFG)
        self.assertLess(cal.score(0.25, 0.02), cal.threshold)


class TestTrustMarkMapping(unittest.TestCase):
    def setUp(self):
        self.reg = fake_registry()
        self.tm = CFG["trustmark"]

    def test_payload_is_sha_prefix(self):
        self.assertEqual(sha256_prefix_bits(self.reg.source_sha256[0]), PAYLOAD_SRC01)

    def test_match(self):
        r = map_trustmark({"detected": True, "payload": PAYLOAD_SRC01, "schema": 1}, self.reg, self.tm)
        self.assertEqual((r["status"], r["source_index"]), ("KODE_COCOK", 0))

    def test_not_detected(self):
        r = map_trustmark({"detected": False, "payload": "", "schema": -1}, self.reg, self.tm)
        self.assertEqual((r["status"], r["source_index"]), ("TIDAK_TERDETEKSI", None))

    def test_wrong_schema(self):
        r = map_trustmark({"detected": True, "payload": "1" * 75, "schema": 3}, self.reg, self.tm)
        self.assertEqual(r["status"], "SCHEMA_TIDAK_SESUAI")

    def test_unknown_code(self):
        r = map_trustmark({"detected": True, "payload": "0" * 61, "schema": 1}, self.reg, self.tm)
        self.assertEqual((r["status"], r["source_index"]), ("KODE_TIDAK_DIKENAL", None))


class TestDecision(unittest.TestCase):
    TH = CFG["decision_threshold"]
    TOP = [{"index": 6, "similarity": 0.9}, {"index": 14, "similarity": 0.4}]

    def test_agree(self):
        d = decide(6, self.TOP, 0.5, self.TH)
        self.assertEqual((d["verdict"], d["rule"], d["source_index"]), ("ACCEPT", "tm_dino_agree", 6))

    def test_conflict_even_with_high_score(self):
        d = decide(11, self.TOP, 0.999, self.TH)
        self.assertEqual((d["verdict"], d["rule"]), ("ABSTAIN", "tm_dino_conflict"))

    def test_tm_missing_accept(self):
        d = decide(None, self.TOP, self.TH, self.TH)
        self.assertEqual((d["verdict"], d["source_index"]), ("ACCEPT", 6))

    def test_tm_missing_abstain(self):
        d = decide(None, self.TOP, self.TH - 1e-9, self.TH)
        self.assertEqual((d["verdict"], d["rule"]), ("ABSTAIN", "tm_missing_score_below_threshold"))

    def test_no_candidates(self):
        self.assertEqual(decide(None, [], 0.99, self.TH)["rule"], "insufficient_evidence")


class TestPrepareAndRank(unittest.TestCase):
    def test_exif_rotation_and_rgb(self):
        img = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
        exif = Image.Exif()
        exif[0x0112] = 6  # putar 90°
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", exif=exif)
        out, info = prepare_image(buf.getvalue())
        self.assertEqual((out.mode, out.size), ("RGB", (20, 40)))
        self.assertEqual(len(info["sha256"]), 64)

    def test_rank_returns_self_first(self):
        reg = fake_registry()
        top = rank_candidates(reg.embeddings[7], reg, k=3)
        self.assertEqual(top[0]["number"], "08")
        self.assertAlmostEqual(top[0]["similarity"], 1.0, places=5)
        self.assertGreater(top[0]["similarity"], top[1]["similarity"])


class TestResearchIds(unittest.TestCase):
    def test_forty_sources(self):
        ids = load_research_ids()
        self.assertEqual(len(ids), 40)
        for s in ids:
            self.assertEqual(s["source_id"], "SRC_" + s["source_sha256"][:16])


if __name__ == "__main__":
    unittest.main()
