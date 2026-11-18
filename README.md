# JATIKARYA – Aplikasi Verifikasi Asal Gambar

Aplikasi web dari penelitian *Ketahanan Watermark TrustMark dan Pencocokan Visual DINOv2 dalam Verifikasi Asal Gambar setelah Pengeditan Generatif*.

Pengguna mengunggah gambar hasil edit AI. Aplikasi memprosesnya dalam enam tahap yang sama dengan penelitian, lalu menyatakan apakah gambar itu **DITERIMA** sebagai turunan salah satu dari 40 foto sumber, atau **DITAHAN** untuk dicek manual.

| Tahap | Isi | Sumber parameter |
|---|---|---|
| 1. Unggah foto | JPG/PNG/WEBP/HEIC, maks. 25 MB | – |
| 2. Persiapan | frame 0 → EXIF transpose → RGB, SHA-256 | `EXIF_TRANSPOSE_RGB_FRAME0` |
| 3. TrustMark | model Q, BCH_5, mode biner; payload 61 bit = 61 bit pertama SHA-256 foto sumber | `config.json` STEP TrustMark |
| 4. DINOv2 | `dinov2_vits14`, Resize 518 → CenterCrop 518 → ImageNet, kosinus ke 40 referensi | `config.json` STEP DINOv2 |
| 5. Calibrated Fusion | regresi logistik (kemiripan top-1, selisih top-1/top-2), ambang 0,8239 | `calibration_protocol_locked.json` |
| 6. Keputusan | TM & DINO sepakat → terima; bertentangan → tahan; TM tidak ada → terima jika skor ≥ ambang | aturan `TM_FIRST_THEN_DINOV2_TOP1` |

## Menjalankan di Google Colab

**Step 1.** Buka `notebooks/JATIKARYA_App_Colab.ipynb` di Colab, lalu pilih *Runtime → Change runtime type → T4 GPU*.

**Step 2.** Jalankan sel *Step 1* untuk menghubungkan Google Drive.

**Step 3.** Jalankan sel *Step 2* untuk mengambil kode dan memasang pustaka.

**Step 4.** Di sel *Step 3*, isi `PHOTOS` dengan folder **40 foto sumber asli** (yang belum diberi watermark).

**Step 5.** Jalankan sel *Step 4*. Pastikan baris `[Step 2]` menampilkan `40/40`. Hasilnya, `registry.npz`, disimpan di Drive, jadi langkah ini cukup sekali.

**Step 6.** Jalankan sel *Step 5*. Server siap kalau muncul `{'ready': True, ...}`.

**Step 7.** Jalankan sel *Step 6*, lalu buka alamat `https://….trycloudflare.com` yang muncul dari HP atau laptop.

**Step 8 (opsional).** Sel *Step 7* menguji satu gambar dari dataset penelitian. Bandingkan hasilnya dengan `final_test_predictions_locked.csv`.

Alamat tunnel berubah setiap kali sesi Colab dimulai ulang.

## Uji logika (tanpa GPU)

```
python -m unittest discover -s tests -v
```

Tes ini memastikan rumus kalibrasi menghasilkan skor yang sama persis dengan `final_test_predictions_locked.csv`, yaitu sampai 10 angka di belakang koma, termasuk kasus paling mepet ambang (skor 0,82452 terhadap ambang 0,82392). Tes juga memeriksa pemetaan payload TrustMark ke foto sumber dan aturan keputusan.

## Struktur

```
app/
  pipeline.py           enam tahap verifikasi
  main.py               server FastAPI (/, /api/health, /api/verify)
  static/index.html     tampilan 6 tahap
  research/
    calibration.json    parameter kalibrasi terkunci + konfigurasi model
    reference_ids.json  40 ID dan hash foto sumber (tanpa foto)
scripts/build_registry.py   membuat registry.npz dari 40 foto sumber
notebooks/JATIKARYA_App_Colab.ipynb
tests/test_logic.py
```

## Batasan

- Hanya mengenali 40 foto sumber penelitian. Foto lain akan berakhir **DITAHAN**, atau bisa salah cocok kalau sangat mirip. Ambang belum diuji untuk gambar dari luar koleksi.
- Ambang kalibrasi bersifat awal (*preliminary*). Data kalibrasi hanya berisi dua prediksi DINOv2 yang salah.
- Foto sumber tidak disimpan di repositori. `registry.npz` hanya berisi hash dan embedding, dan sebaiknya tetap disimpan di Drive.
