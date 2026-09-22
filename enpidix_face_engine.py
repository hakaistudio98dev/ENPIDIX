"""
enpidix_face_engine.py
======================================================================
ENPIDIX VMS — Mesin pengenalan wajah TAHAN OKLUSI (kerudung / topi / masker).

KENAPA VERSI LAMA GAGAL
-----------------------
1) TAHAP DETEKSI — `haarcascade_frontalface_default.xml` dilatih pakai wajah
   frontal UTUH. Masker menutup hidung+mulut, topi menutup dahi + bikin bayangan
   keras di mata, kerudung mengubah siluet kepala. Ketiganya bikin cascade GAGAL
   MENEMUKAN wajah sama sekali. Kalau wajah tidak terdeteksi, secanggih apapun
   recognizer-nya tidak akan pernah dipanggil.

2) TAHAP PENGENALAN — LBPH membandingkan histogram tekstur SELURUH kotak wajah
   dalam grid 8x8. Masker mengganti ~45% sel grid dengan kain polos, topi
   mengganti baris atas. Jarak chi-square langsung meledak jauh di atas
   threshold 70 -> selalu "Tidak Dikenal".

3) TIDAK ADA ALIGNMENT — LBPH itu berbasis grid, jadi kepala miring 10 derajat
   (sangat umum pada pemakai kerudung) menggeser seluruh grid dan merusak
   pencocokan.

4) SAMPEL TRAINING BERSIH SEMUA — yang didaftarkan cuma foto wajah terbuka,
   jadi model tidak pernah "melihat" versi bermasker/bertopi/berkerudung.

SOLUSI DI MODUL INI (5 lapis)
-----------------------------
L1 DETEKTOR   : YuNet (cv2.FaceDetectorYN, ONNX ~230KB) dilatih di WIDER FACE
                yang penuh wajah terhalang -> masker/topi/kerudung tetap
                ketemu. Fallback berjenjang ke Haar frontal + Haar profil.
L2 ALIGNMENT  : Rotasi & skala pakai 2 landmark mata dari YuNet supaya mata
                selalu di koordinat kanonik yang sama. + CLAHE (bukan
                equalizeHist) supaya bayangan lidah topi tidak menghapus tekstur.
L3 ENSEMBLE   : 5 model LBPH per-REGION, bukan 1 model wajah penuh:
                  full   -> wajah utuh          (kondisi normal)
                  upper  -> dahi+alis+mata      (TAHAN MASKER)
                  eyes   -> pita mata saja      (TAHAN MASKER + TOPI + KERUDUNG)
                  lower  -> hidung+mulut+dagu   (TAHAN TOPI + KERUDUNG)
                  core   -> oval tengah         (TAHAN KERUDUNG; buang rambut/telinga)
                Deteksi oklusi murah (edge-density) memilih region mana yang
                dipercaya, sisanya dibobot turun. Skor difusikan.
L4 AUGMENTASI : Saat registrasi, 1 foto bersih otomatis diperbanyak jadi ~16
                sampel termasuk versi MASKER, TOPI, dan KERUDUNG sintetis +
                blur (simulasi kamera CCTV) + rotasi + gamma. Ini lompatan
                akurasi terbesar tanpa minta user foto ulang.
L5 VOTING     : Label diputuskan dari voting beberapa frame per-track, bukan
                dari 1 frame. Menghilangkan salah-nama saat wajah terhalang.

OPSIONAL (SANGAT DIREKOMENDASIKAN) — SFace embedding
    Kalau file face_recognition_sface_2021dec.onnx (~37MB) tersedia, engine
    otomatis pakai embedding 128-dimensi + cosine similarity. Jauh lebih tahan
    oklusi/pose daripada LBPH, DAN biayanya konstan berapapun jumlah orang
    terdaftar (LBPH makin lambat makin banyak sampel). Embedding juga
    didaftarkan untuk versi masker/topi/kerudung sintetis.
    Kalau file-nya tidak ada, engine jalan normal pakai ensemble LBPH.

Lisensi model: Apache-2.0 (OpenCV Zoo). Aman untuk dipakai komersial.
======================================================================
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

# =====================================================================
# KONFIGURASI (semua bisa dioverride lewat environment variable)
# =====================================================================

FACE_MODEL_DIR = os.environ.get("FACE_MODEL_DIR", "models")
YUNET_PATH = os.environ.get(
    "YUNET_MODEL",
    os.path.join(FACE_MODEL_DIR, "face_detection_yunet_2023mar.onnx"),
)
SFACE_PATH = os.environ.get(
    "SFACE_MODEL",
    os.path.join(FACE_MODEL_DIR, "face_recognition_sface_2021dec.onnx"),
)

# Ukuran kanonik hasil alignment. Semua crop wajah (training MAUPUN prediksi)
# wajib melewati align_face() sehingga skala & posisi mata selalu identik.
CANON_W, CANON_H = 200, 200
# Posisi target mata di gambar kanonik (fraksi dari lebar/tinggi).
EYE_TARGET_L = (0.345, 0.395)   # mata kiri subjek (tampak di kanan gambar)
EYE_TARGET_R = (0.655, 0.395)   # mata kanan subjek

# Ambang skor deteksi YuNet. Diturunkan dari default 0.9 -> 0.55 karena wajah
# yang tertutup masker/kerudung memang wajar dapat skor lebih rendah.
YUNET_SCORE = float(os.environ.get("YUNET_SCORE_THRESHOLD", "0.55"))
YUNET_NMS = float(os.environ.get("YUNET_NMS_THRESHOLD", "0.3"))
YUNET_TOPK = int(os.environ.get("YUNET_TOP_K", "50"))
# Resolusi kerja detektor. YuNet sangat cepat; 320px cukup untuk wajah >48px.
YUNET_INPUT_W = int(os.environ.get("YUNET_INPUT_WIDTH", "320"))

# Mode engine: "auto" | "sface" | "lbph"
ENGINE_MODE = os.environ.get("FACE_ENGINE_MODE", "auto").lower()

# --- Skoring LBPH: RELATIF, bukan ambang absolut --------------------------
# Nilai jarak chi-square LBPH sangat bergantung kamera, pencahayaan, dan ukuran
# region. Uji internal: satu wajah yang SAMA persis menghasilkan jarak 0 saat
# gambarnya identik, tapi 59-87 setelah diberi blur+derau ala CCTV — padahal
# orangnya benar dan tetap peringkat 1 di semua region. Ambang absolut (dulu 70)
# jadi mustahil ditentukan sekali untuk semua lokasi: kalau ketat, orang asli
# ditolak; kalau longgar, orang lain diterima.
#
# Karena itu penilaian dibuat RELATIF terhadap sebaran jarak ke SEMUA orang
# terdaftar pada frame itu juga (lewat predict_collect). Wajah asli selalu jatuh
# jauh di bawah jarak khas orang-lain, berapapun nilai mutlaknya. Ambang jadi
# mengkalibrasi dirinya sendiri di tiap kamera tanpa perlu ditune manual.
#
#   z = (jarak_khas_orang_lain - jarak_kandidat) / jarak_khas_orang_lain
#   z > 0  -> lebih mirip daripada orang lain pada umumnya
#   z = (jarak_khas_orang_lain - jarak_kandidat) / sebaran_jarak_orang_lain
# Pembaginya adalah SEBARAN (MAD), bukan nilai tengahnya. Ini penting: dengan
# pembagi nilai tengah, kasus masker gagal karena semua jarak saling berdekatan
# (86 vs median 90) sehingga z ikut mengecil padahal orangnya jelas peringkat 1.
# Dengan pembagi sebaran, yang diukur adalah "berapa kali lipat lebih menonjol
# kandidat ini dibanding variasi normal antar orang lain" — tetap tajam walau
# nilai mutlak jaraknya berdempetan.
LBPH_HARD_MAX = float(os.environ.get("LBPH_HARD_MAX", "145"))   # pagar pengaman saja
Z_REF = float(os.environ.get("FACE_Z_REF", "3.0"))   # z sebesar ini dianggap yakin penuh
# Ambang cosine SFace. Rekomendasi resmi OpenCV 0.363 untuk wajah terbuka.
# Untuk wajah terhalang diturunkan sedikit, tapi selalu ditemani uji MARGIN.
SFACE_COSINE = float(os.environ.get("SFACE_COSINE_THRESHOLD", "0.32"))

# Skor gabungan minimal supaya sebuah nama diterima.
# Nilai default di bawah berasal dari sapuan ambang pada uji internal
# (20 identitas terdaftar vs 60 orang asing, semua dengan blur+derau ala CCTV):
#   SIM 0.40 / margin 0.15 -> dikenali 100%, tapi orang ASING salah-terima 16.7%
#   SIM 0.75 / margin 0.28 -> dikenali  98%, orang asing salah-terima ~1%
# Untuk sistem keamanan, salah-terima jauh lebih mahal daripada sesekali harus
# menunggu satu-dua frame lagi, jadi dipilih titik kerja yang kedua.
ACCEPT_SIM = float(os.environ.get("FACE_ACCEPT_SIM", "0.75"))
# Selisih minimal antara kandidat terbaik dan kandidat terbaik dari ORANG LAIN.
# Ini kunci anti-salah-sebut-nama saat wajah terhalang: kalau dua orang sama-sama
# "lumayan mirip", lebih baik jawab Tidak Dikenal daripada salah menyebut nama.
ACCEPT_MARGIN = float(os.environ.get("FACE_ACCEPT_MARGIN", "0.28"))

# Voting temporal
VOTE_WINDOW = int(os.environ.get("FACE_VOTE_WINDOW", "5"))
VOTE_MIN_AGREE = int(os.environ.get("FACE_VOTE_MIN_AGREE", "3"))

# Augmentasi
AUG_PER_PHOTO = int(os.environ.get("FACE_AUG_PER_PHOTO", "24"))
AUG_ENABLED = os.environ.get("FACE_AUG_ENABLED", "1") != "0"
# Batas jumlah template LBPH per region. LBPH.predict() itu O(jumlah_template),
# jadi tanpa batas ini server bisa melambat drastis setelah puluhan pendaftaran.
# Batas jumlah template LBPH per region. predict() itu O(jumlah_template):
# pada uji, 900 template x 5 region = ~45 ms per wajah di 1 core — terlalu berat
# untuk server multi-kamera bertenaga rendah. 500 menekannya ke ~25 ms.
# (Mode SFace tidak terpengaruh sama sekali: biayanya tetap walau ribuan orang.)
LBPH_MAX_TEMPLATES = int(os.environ.get("LBPH_MAX_TEMPLATES", "500"))

# Ambang heuristik deteksi oklusi (silakan tune lewat /api/faces/engine/debug)
OCC_MASK_RATIO = float(os.environ.get("OCC_MASK_RATIO", "0.55"))
OCC_HEADWEAR_RATIO = float(os.environ.get("OCC_HEADWEAR_RATIO", "0.55"))

# Maksimum wajah yang diproses recognition per frame (jaga CPU).
MAX_FACES_PER_FRAME = int(os.environ.get("FACE_MAX_PER_FRAME", "5"))


# =====================================================================
# DEFINISI REGION
#   (x0, y0, x1, y1) dalam fraksi gambar kanonik, plus ukuran output LBPH.
#   Ukuran output sengaja dibuat kecil-kecil supaya train & predict tetap
#   ringan di CPU lemah (MacBook Air 2017 / Jetson tanpa CUDA).
# =====================================================================
REGIONS = {
    #                x0     y0     x1     y1      out(w,h)   tahan terhadap
    "full":  ((0.00, 0.00, 1.00, 1.00), (112, 112)),   # kondisi normal
    "upper": ((0.05, 0.02, 0.95, 0.56), (112, 68)),    # MASKER
    "eyes":  ((0.06, 0.22, 0.94, 0.58), (120, 44)),    # MASKER + TOPI + KERUDUNG
    "lower": ((0.10, 0.46, 0.90, 0.98), (104, 60)),    # TOPI + KERUDUNG
    "core":  ((0.18, 0.16, 0.82, 0.92), (88, 104)),    # KERUDUNG
}

# Bobot dasar tiap region saat TIDAK ada oklusi terdeteksi.
BASE_WEIGHTS = {"full": 1.00, "upper": 0.55, "eyes": 0.45, "lower": 0.55, "core": 0.70}

# Bobot saat kondisi oklusi tertentu terdeteksi. Region yang tertutup dibuat 0
# supaya sama sekali tidak mencemari skor.
# Bobot saat kondisi oklusi tertentu terdeteksi. Sengaja TIDAK ADA yang 0:
# deteksi oklusi itu heuristik dan bisa salah tebak (mis. rambut gelap terbaca
# sebagai topi). Kalau gerbangnya biner, satu salah tebak langsung mematikan
# region yang sebenarnya paling informatif. Jadi ini hanya PRIOR lembut —
# penyaringan sesungguhnya dikerjakan oleh seleksi top-K di _fuse_regions().
OCC_WEIGHTS = {
    "mask":          {"full": 0.30, "upper": 1.00, "eyes": 0.95, "lower": 0.15, "core": 0.55},
    "headwear":      {"full": 0.35, "upper": 0.40, "eyes": 0.90, "lower": 1.00, "core": 0.95},
    "mask_headwear": {"full": 0.20, "upper": 0.55, "eyes": 1.00, "lower": 0.20, "core": 0.55},
}

# Berapa region terbaik yang dipakai untuk menilai tiap kandidat.
# Alasannya: region yang tertutup kain menghasilkan z mendekati nol untuk SEMUA
# orang (tidak membedakan siapa-siapa). Kalau dirata-rata bersama region yang
# sehat, region mati itu justru MENGENCERKAN bukti yang valid. Dengan mengambil
# beberapa region terbaik saja, region yang tertutup tersingkir dengan
# sendirinya — tanpa perlu deteksi oklusi menebak dengan benar.
FUSE_TOP_K = int(os.environ.get("FACE_FUSE_TOP_K", "3"))

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Redam derau sensor SEBELUM CLAHE.

    LBP itu operator perbandingan antar-piksel tetangga. Derau sensor sebesar
    +/-6 level saja sudah cukup membalik bit LBP di seluruh wajah — pada uji
    internal skor kecocokan jatuh dari 1.00 ke 0.28 hanya karena derau, padahal
    wajahnya sama persis. Kamera CCTV malam hari justru paling berderau, jadi
    langkah ini wajib. Gaussian 3x3 cukup: menghapus frekuensi tinggi tanpa
    memakan tekstur kulit yang justru jadi ciri identitas.
    CLAHE dipasang SESUDAHNYA supaya penguatan kontras tidak ikut memperkuat derau.
    """
    return cv2.GaussianBlur(gray, (3, 3), 0)


# =====================================================================
# UTILITAS GAMBAR
# =====================================================================

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_gray(face_gray: np.ndarray) -> np.ndarray:
    """Samakan skala + ratakan kontras lokal.

    Pakai CLAHE, BUKAN cv2.equalizeHist. equalizeHist bersifat global: satu
    bayangan gelap besar dari lidah topi menggeser seluruh histogram dan
    menghapus tekstur di area yang sebenarnya masih terlihat. CLAHE bekerja
    per-tile 8x8 sehingga area terang tetap detail walau ada area gelap pekat
    di sebelahnya — persis kasus topi & kerudung.
    """
    if face_gray is None or face_gray.size == 0:
        raise ValueError("Crop wajah kosong.")
    g = _to_gray(face_gray)
    resized = cv2.resize(g, (CANON_W, CANON_H), interpolation=cv2.INTER_LINEAR)
    return _CLAHE.apply(_denoise(resized))


def crop_region(canon_gray: np.ndarray, region: str) -> np.ndarray:
    (x0, y0, x1, y1), out = REGIONS[region]
    h, w = canon_gray.shape[:2]
    sub = canon_gray[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if sub.size == 0:
        sub = canon_gray
    return cv2.resize(sub, out, interpolation=cv2.INTER_LINEAR)


# =====================================================================
# ALIGNMENT
# =====================================================================

def align_from_eyes(img: np.ndarray, eye_r: tuple, eye_l: tuple) -> np.ndarray:
    """Warp affine supaya kedua mata mendarat tepat di EYE_TARGET_*.

    Ini menormalkan ROTASI (kepala miring), SKALA (jauh/dekat kamera), dan
    TRANSLASI sekaligus. Tanpa ini, grid LBPH bergeser dan pencocokan hancur —
    penyebab paling sering "sudah terdaftar tapi tidak dikenali".
    eye_r/eye_l dalam koordinat piksel gambar sumber (right eye / left eye subjek).
    """
    src = np.float32([eye_r, eye_l, [0, 0]])
    tgt_r = np.float32([EYE_TARGET_R[0] * CANON_W, EYE_TARGET_R[1] * CANON_H])
    tgt_l = np.float32([EYE_TARGET_L[0] * CANON_W, EYE_TARGET_L[1] * CANON_H])

    # Bangun similarity transform (rotasi+skala+translasi) dari 2 titik.
    dx, dy = eye_l[0] - eye_r[0], eye_l[1] - eye_r[1]
    dist = float(np.hypot(dx, dy))
    if dist < 1e-3:
        raise ValueError("Jarak antar mata tidak valid.")
    tgt_dist = float(np.hypot(tgt_l[0] - tgt_r[0], tgt_l[1] - tgt_r[1]))
    scale = tgt_dist / dist
    angle = np.degrees(np.arctan2(dy, dx))

    center = ((eye_r[0] + eye_l[0]) / 2.0, (eye_r[1] + eye_l[1]) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += (tgt_r[0] + tgt_l[0]) / 2.0 - center[0]
    M[1, 2] += (tgt_r[1] + tgt_l[1]) / 2.0 - center[1]

    _ = src  # (disimpan untuk keterbacaan; transform dibangun analitik di atas)
    return cv2.warpAffine(img, M, (CANON_W, CANON_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def align_from_box(img: np.ndarray, box) -> np.ndarray:
    """Fallback tanpa landmark: perluas kotak sedikit lalu resize ke kanonik.

    Kotak Haar biasanya terlalu ketat di dagu dan memotong dahi. Kita lebarkan
    agar proporsinya mendekati hasil alignment berbasis mata.
    """
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * 1.18
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side * 0.56))
    x1 = int(min(img.shape[1], cx + side / 2))
    y1 = int(min(img.shape[0], cy + side * 0.62))
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        raise ValueError("Crop di luar frame.")
    return cv2.resize(sub, (CANON_W, CANON_H), interpolation=cv2.INTER_LINEAR)


# =====================================================================
# AUGMENTASI OKLUSI SINTETIS
#   Inti dari "1 foto bersih -> model paham versi bermasker/bertopi/berkerudung".
# =====================================================================

def _apply_mask(canon: np.ndarray, tone: int = 205, low: bool = False) -> np.ndarray:
    """Tempel masker sintetis: poligon menutup hidung-mulut-dagu."""
    out = canon.copy()
    h, w = out.shape[:2]
    top = 0.50 if low else 0.545
    pts = np.array([
        [0.09 * w, top * h], [0.50 * w, (top - 0.045) * h], [0.91 * w, top * h],
        [0.88 * w, 0.80 * h], [0.50 * w, 1.02 * h], [0.12 * w, 0.80 * h],
    ], dtype=np.int32)

    layer = np.full_like(out, tone)
    # gradasi vertikal tipis + garis lipatan supaya tidak jadi bidang datar total
    grad = np.linspace(-14, 14, h, dtype=np.float32).reshape(-1, 1)
    layer = np.clip(layer.astype(np.float32) + grad, 0, 255).astype(np.uint8)
    cv2.line(layer, (int(0.10 * w), int(0.70 * h)), (int(0.90 * w), int(0.70 * h)),
             int(max(0, tone - 30)), 2)

    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255)
    out[m > 0] = layer[m > 0]
    return out


def _apply_cap(canon: np.ndarray, depth: float = 0.26, tone: int = 55) -> np.ndarray:
    """Tempel topi sintetis: pita gelap di atas + bayangan lidah topi di bawahnya."""
    out = canon.copy()
    h, w = out.shape[:2]
    cut = int(depth * h)
    out[:cut, :] = tone
    cv2.line(out, (0, cut), (w, cut), max(0, tone - 25), 2)
    # bayangan lidah topi: gelapkan bertahap sampai sekitar alis
    sh_end = int(min(h, cut + 0.11 * h))
    if sh_end > cut:
        band = out[cut:sh_end, :].astype(np.float32)
        ramp = np.linspace(0.55, 1.0, sh_end - cut, dtype=np.float32).reshape(-1, 1)
        out[cut:sh_end, :] = np.clip(band * ramp, 0, 255).astype(np.uint8)
    return out


def _apply_hijab(canon: np.ndarray, tight: bool = False, tone: int = 105) -> np.ndarray:
    """Tempel kerudung sintetis: sisakan hanya oval wajah, sisanya kain.

    Ini mensimulasikan hilangnya rambut, telinga, garis rahang luar, dan leher —
    justru fitur-fitur itulah yang selama ini dipakai LBPH dan yang hilang saat
    orang memakai kerudung.
    """
    out = canon.copy()
    h, w = out.shape[:2]
    ax = int((0.30 if tight else 0.355) * w)
    ay = int((0.42 if tight else 0.465) * h)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, int(0.56 * h)), (ax, ay), 0, 0, 360, 255, -1)
    m[: int(0.13 * h), :] = 0          # garis kerudung di batas dahi
    fabric = np.full_like(out, tone)
    cv2.line(fabric, (0, int(0.30 * h)), (w, int(0.55 * h)), max(0, tone - 25), 3)
    out[m == 0] = fabric[m == 0]
    return out


def _rotate(canon: np.ndarray, deg: float) -> np.ndarray:
    h, w = canon.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(canon, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _blurify(canon: np.ndarray, factor: float = 0.45) -> np.ndarray:
    """Simulasi wajah kecil di stream CCTV: turunkan resolusi lalu besarkan lagi.

    Foto pendaftaran biasanya tajam (kamera HP), sementara wajah dari CCTV
    seringkali cuma 60-80px dan lembek. Selisih ketajaman ini saja sudah cukup
    membuat pola LBP berbeda. Augmentasi ini menutup gap tersebut.
    """
    h, w = canon.shape[:2]
    small = cv2.resize(canon, (max(8, int(w * factor)), max(8, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _gamma(canon: np.ndarray, g: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(canon, lut)


def _noise(canon: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    r = np.random.RandomState(seed)
    n = r.normal(0, sigma, canon.shape).astype(np.float32)
    return np.clip(canon.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _shift(canon: np.ndarray, dx: int, dy: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(canon, M, (canon.shape[1], canon.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


# Varian OKLUSI — mengajari model seperti apa orang ini saat memakai
# masker / topi / kerudung, walau yang didaftarkan cuma foto wajah terbuka.
_OCCLUSIONS = [
    ("bersih",          lambda a: a),
    ("masker_terang",   lambda a: _apply_mask(a, 205)),
    ("masker_gelap",    lambda a: _apply_mask(a, 80, low=True)),
    ("masker_sedang",   lambda a: _apply_mask(a, 150)),
    ("topi",            lambda a: _apply_cap(a, 0.26, 55)),
    ("topi_rendah",     lambda a: _apply_cap(a, 0.32, 35)),
    ("kerudung",        lambda a: _apply_hijab(a, False)),
    ("kerudung_ketat",  lambda a: _apply_hijab(a, True)),
    ("kerudung_masker", lambda a: _apply_mask(_apply_hijab(a, False), 205)),
    ("topi_masker",     lambda a: _apply_mask(_apply_cap(a, 0.26, 55), 205)),
]

# Varian KONDISI TANGKAP — menutup jurang antara foto pendaftaran (tajam, terang,
# lurus, dari HP) dan wajah dari CCTV (buram, berderau, miring, tergeser).
_PERTURBS = [
    ("apa_adanya", lambda a, k: a),
    ("buram",      lambda a, k: _blurify(a, 0.5)),
    ("derau",      lambda a, k: _noise(a, 7, k)),
    ("miring_kanan", lambda a, k: _rotate(a, 7)),
    ("miring_kiri",  lambda a, k: _rotate(a, -7)),
    ("geser",      lambda a, k: _shift(a, 4, -3)),
    ("geser2",     lambda a, k: _shift(a, -4, 3)),
    ("redup",      lambda a, k: _gamma(a, 1.35)),
    ("terang",     lambda a, k: _gamma(a, 0.72)),
    ("buram_derau", lambda a, k: _noise(_blurify(a, 0.5), 6, k)),
]


def augment_canonical(canon: np.ndarray, limit: int = AUG_PER_PHOTO) -> list:
    """1 wajah kanonik bersih -> banyak sampel latih yang meniru kondisi lapangan.

    Dibangun sebagai perkalian silang OKLUSI x KONDISI TANGKAP, diselang-seling
    supaya dengan kuota kecil pun setiap jenis oklusi tetap kebagian beberapa
    kondisi tangkap yang berbeda. Deterministik (bukan acak) supaya hasil
    training bisa direproduksi dan didebug.
    """
    if not AUG_ENABLED:
        return [canon]

    flip = cv2.flip(canon, 1)
    out, seen = [], set()
    limit = max(1, limit)

    idx = 0
    for round_no in range(len(_PERTURBS)):
        for o_i, (_, occ_fn) in enumerate(_OCCLUSIONS):
            if len(out) >= limit:
                break
            p_name, pert_fn = _PERTURBS[(o_i + round_no) % len(_PERTURBS)]
            base = flip if ((o_i + round_no) % 4 == 3) else canon
            try:
                img = pert_fn(occ_fn(base), idx)
            except Exception:
                continue
            idx += 1
            key = hash(img.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(img)
        if len(out) >= limit:
            break
    return out or [canon]


# =====================================================================
# DETEKSI OKLUSI (murah, tanpa model tambahan)
# =====================================================================

def _edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(gx, gy)))


def analyze_occlusion(canon_gray: np.ndarray) -> dict:
    """Tebak apakah bagian bawah (masker) / bagian atas (topi atau kerudung)
    tertutup, dengan membandingkan KEPADATAN TEPI tiap pita terhadap pita mata.

    Logikanya sederhana dan tahan banting: kain itu polos. Hidung, mulut, dan
    bibir menghasilkan banyak tepi; masker menghasilkan hampir nol. Dahi
    berambut/berkerut punya tepi; kain kerudung dan pita topi tidak. Pita mata
    dipakai sebagai referensi karena itu satu-satunya area yang TIDAK PERNAH
    tertutup oleh ketiga kondisi tersebut.
    """
    h, w = canon_gray.shape[:2]
    eyes = canon_gray[int(0.26 * h):int(0.50 * h), int(0.12 * w):int(0.88 * w)]
    lower = canon_gray[int(0.60 * h):int(0.94 * h), int(0.18 * w):int(0.82 * w)]
    fore = canon_gray[int(0.03 * h):int(0.22 * h), int(0.18 * w):int(0.82 * w)]

    e_eyes = _edge_density(eyes) + 1e-6
    r_low = _edge_density(lower) / e_eyes
    r_fore = _edge_density(fore) / e_eyes

    mask = r_low < OCC_MASK_RATIO
    headwear = r_fore < OCC_HEADWEAR_RATIO

    if mask and headwear:
        state = "mask_headwear"
    elif mask:
        state = "mask"
    elif headwear:
        state = "headwear"
    else:
        state = "none"

    return {
        "state": state, "mask": bool(mask), "headwear": bool(headwear),
        "ratio_lower": round(r_low, 3), "ratio_forehead": round(r_fore, 3),
        "edge_eyes": round(e_eyes, 2),
    }


# =====================================================================
# ENGINE
# =====================================================================

def _standout(d: float, others: list) -> float:
    """Seberapa menonjol jarak `d` dibanding sebaran jarak ke orang-orang lain.

    Dinyatakan dalam satuan sebaran (MAD, robust terhadap pencilan), bukan dalam
    satuan mutlak. Konsekuensinya ambang tidak perlu ditune ulang tiap kamera:
    kalau seluruh jarak naik gara-gara ruangan gelap, pembaginya ikut naik.
    """
    if not others:
        # Cuma satu orang terdaftar -> tidak ada pembanding. Jatuh balik ke
        # penilaian mutlak yang konservatif terhadap pagar pengaman.
        return max(0.0, (LBPH_HARD_MAX - d) / max(1e-6, LBPH_HARD_MAX)) * Z_REF
    arr = np.asarray(others, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    sd = max(mad, 0.02 * med, 1e-6)   # lantai: cegah pembagian oleh sebaran ~0
    return max(0.0, (med - d) / sd)


def _fuse_regions(contribs: list) -> float:
    """Gabungkan bukti dari beberapa region jadi satu skor, memakai K region
    TERBAIK saja (lihat catatan pada FUSE_TOP_K)."""
    if not contribs:
        return 0.0
    top = sorted(contribs, key=lambda zw: zw[0], reverse=True)[:max(1, FUSE_TOP_K)]
    wsum = sum(w for _, w in top)
    if wsum <= 0:
        return 0.0
    return sum(z * w for z, w in top) / wsum


class FaceEngine:
    def __init__(self):
        self._local = threading.local()          # detektor per-thread (YuNet stateful)
        self._model_lock = threading.RLock()
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.haar_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml")

        self.yunet_available = os.path.exists(YUNET_PATH)
        self.sface_available = os.path.exists(SFACE_PATH)
        if ENGINE_MODE == "lbph":
            self.sface_available = False

        # state model
        self.lbph = {}          # region -> LBPHFaceRecognizer
        self.lbph_ready = {}    # region -> bool
        self.id_to_name = {}
        self.embeddings = []    # list[(person_id, np.ndarray)]
        self.stats = {"persons": 0, "photos": 0, "samples": 0, "embeddings": 0,
                      "trained_at": None, "train_seconds": 0.0}

    # ---------- detektor per-thread ----------
    def _yunet(self, w: int, h: int):
        if not self.yunet_available:
            return None
        d = getattr(self._local, "yunet", None)
        if d is None:
            try:
                d = cv2.FaceDetectorYN.create(
                    YUNET_PATH, "", (w, h), YUNET_SCORE, YUNET_NMS, YUNET_TOPK)
            except Exception:
                self.yunet_available = False
                return None
            self._local.yunet = d
            self._local.yunet_size = (w, h)
        if getattr(self._local, "yunet_size", None) != (w, h):
            d.setInputSize((w, h))
            self._local.yunet_size = (w, h)
        return d

    def _sface(self):
        if not self.sface_available:
            return None
        r = getattr(self._local, "sface", None)
        if r is None:
            try:
                r = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
            except Exception:
                self.sface_available = False
                return None
            self._local.sface = r
        return r

    # ---------- DETEKSI ----------
    def detect(self, frame_bgr: np.ndarray) -> list:
        """Kembalikan list dict: {box:(x,y,w,h), landmarks|None, score, source}.

        Berjenjang: YuNet -> Haar frontal (parameter dilonggarkan) -> Haar profil.
        Yang dilonggarkan pada fallback Haar: minNeighbors 5 -> 3. Wajah
        bermasker/berkerudung menghasilkan lebih sedikit window positif, jadi
        syarat 5 tetangga terlalu ketat dan membuang deteksi yang sebenarnya benar.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        H, W = frame_bgr.shape[:2]
        dets = []

        # --- YuNet ---
        if self.yunet_available:
            scale = min(1.0, YUNET_INPUT_W / float(W))
            sw, sh = max(32, int(W * scale)), max(32, int(H * scale))
            small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR) \
                if scale < 0.999 else frame_bgr
            d = self._yunet(sw, sh)
            if d is not None:
                try:
                    _, faces = d.detect(small)
                except Exception:
                    faces = None
                if faces is not None:
                    inv = 1.0 / scale if scale > 0 else 1.0
                    for f in faces:
                        x, y, w, h = [float(v) * inv for v in f[0:4]]
                        lm = [(float(f[4 + i * 2]) * inv, float(f[5 + i * 2]) * inv)
                              for i in range(5)]
                        dets.append({
                            "box": (int(x), int(y), int(w), int(h)),
                            "landmarks": lm, "score": float(f[14]),
                            "raw": np.array(f, dtype=np.float32) * 1.0,
                            "raw_scale": inv, "source": "yunet",
                        })

        # --- Fallback Haar (hanya kalau YuNet tidak ada / tidak menemukan apapun) ---
        if not dets:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = _CLAHE.apply(gray)
            found = self.haar.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
            if len(found) == 0 and self.haar_profile is not None:
                found = self.haar_profile.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
            for (x, y, w, h) in found:
                dets.append({"box": (int(x), int(y), int(w), int(h)),
                             "landmarks": None, "score": 0.5, "source": "haar"})

        dets.sort(key=lambda d: d["box"][2] * d["box"][3], reverse=True)
        return dets[:MAX_FACES_PER_FRAME]

    # ---------- ALIGN ----------
    def align(self, frame_bgr: np.ndarray, det: dict) -> np.ndarray:
        """Hasilkan wajah kanonik GRAYSCALE 200x200 siap pakai (sudah CLAHE)."""
        gray = _to_gray(frame_bgr)
        lm = det.get("landmarks")
        if lm:
            try:
                canon = align_from_eyes(gray, lm[0], lm[1])
                return _CLAHE.apply(_denoise(canon))
            except Exception:
                pass
        canon = align_from_box(gray, det["box"])
        return _CLAHE.apply(_denoise(cv2.resize(canon, (CANON_W, CANON_H))))

    def align_color(self, frame_bgr: np.ndarray, det: dict) -> np.ndarray:
        lm = det.get("landmarks")
        if lm:
            try:
                return align_from_eyes(frame_bgr, lm[0], lm[1])
            except Exception:
                pass
        return align_from_box(frame_bgr, det["box"])

    # ---------- ENROLL / TRAIN ----------
    def _embed(self, frame_bgr: np.ndarray, det: Optional[dict],
               canon_gray: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        rec = self._sface()
        if rec is None:
            return None
        try:
            if canon_gray is not None:
                src = cv2.cvtColor(canon_gray, cv2.COLOR_GRAY2BGR)
                # SFace minta 112x112 hasil alignCrop; gambar kanonik kita sudah
                # ter-align, jadi cukup diskalakan ke 112x112.
                src = cv2.resize(src, (112, 112), interpolation=cv2.INTER_LINEAR)
                return rec.feature(src).flatten().astype(np.float32)
            if det is not None and det.get("source") == "yunet":
                row = det["raw"].reshape(1, -1)
                row = row.copy()
                row[0, 0:14] *= det.get("raw_scale", 1.0)
                cropped = rec.alignCrop(frame_bgr, row[0])
                return rec.feature(cropped).flatten().astype(np.float32)
        except Exception:
            return None
        return None

    def train(self, rows) -> dict:
        """rows: iterable of (person_id, name, image_path).

        Tiap foto -> align kanonik -> augmentasi oklusi -> masuk ke 5 model LBPH
        per-region DAN (kalau tersedia) jadi embedding SFace.
        """
        t0 = time.time()
        per_region = {r: {"x": [], "y": []} for r in REGIONS}
        embeds, names = [], {}
        photos = 0

        for p_id, name, img_path in rows:
            if not img_path or not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Foto tersimpan kadang sudah berupa crop wajah (alur lama), kadang
            # foto penuh (upload baru). Coba deteksi dulu; kalau tidak ketemu
            # anggap file itu memang sudah crop wajah.
            dets = self.detect(img)
            if dets:
                canon = self.align(img, dets[0])
            else:
                try:
                    canon = normalize_gray(img)
                except Exception:
                    continue

            names[p_id] = name
            photos += 1
            for aug in augment_canonical(canon):
                for r in REGIONS:
                    per_region[r]["x"].append(crop_region(aug, r))
                    per_region[r]["y"].append(p_id)
                e = self._embed(None, None, canon_gray=aug)
                if e is not None:
                    embeds.append((p_id, e / (np.linalg.norm(e) + 1e-9)))

        # Batasi jumlah template LBPH: predict() itu linear terhadap jumlah
        # template, jadi tanpa ini stream jadi tersendat setelah banyak orang.
        new_lbph, new_ready = {}, {}
        for r, data in per_region.items():
            X, Y = data["x"], data["y"]
            if len(X) > LBPH_MAX_TEMPLATES:
                idx = np.linspace(0, len(X) - 1, LBPH_MAX_TEMPLATES).astype(int)
                X = [X[i] for i in idx]
                Y = [Y[i] for i in idx]
            model = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8)
            if X:
                model.train(X, np.array(Y))
                new_ready[r] = True
            else:
                new_ready[r] = False
            new_lbph[r] = model

        with self._model_lock:
            self.lbph, self.lbph_ready = new_lbph, new_ready
            self.id_to_name = names
            self.embeddings = embeds
            self.stats = {
                "persons": len(set(names.values())),
                "photos": photos,
                "samples": len(per_region["full"]["x"]),
                "embeddings": len(embeds),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_seconds": round(time.time() - t0, 2),
            }
        return dict(self.stats)

    # ---------- RECOGNIZE ----------
    def recognize(self, canon_gray: np.ndarray, embed: Optional[np.ndarray] = None,
                  detail: bool = False) -> dict:
        """Kembalikan {name, score, occlusion, mode, per_region}.

        name == None berarti Tidak Dikenal.
        """
        occ = analyze_occlusion(canon_gray)
        weights = OCC_WEIGHTS.get(occ["state"], BASE_WEIGHTS)

        # skor per person_id, 0..1
        scores = {}
        per_region_dbg = {}

        with self._model_lock:
            lbph, ready = self.lbph, self.lbph_ready
            id_to_name, embeddings = self.id_to_name, self.embeddings

        # --- Jalur 1: SFace embedding (kalau ada) ---
        used_sface = False
        if embeddings and embed is not None:
            used_sface = True
            e = embed / (np.linalg.norm(embed) + 1e-9)
            best_per_name = {}
            for p_id, ref in embeddings:
                nm = id_to_name.get(p_id)
                if nm is None:
                    continue
                cos = float(np.dot(e, ref))
                if cos > best_per_name.get(nm, -1.0):
                    best_per_name[nm] = cos
            for nm, cos in best_per_name.items():
                # cosine dipetakan ke skala yang sama dengan z LBPH: 0 tepat di
                # ambang, positif kalau lebih mirip dari ambang.
                sim = (cos - SFACE_COSINE) / max(1e-6, 0.45 - SFACE_COSINE)
                scores[nm] = max(scores.get(nm, 0.0), max(0.0, min(1.0, sim)))
            per_region_dbg["sface"] = {k: round(v, 3) for k, v in best_per_name.items()}

        # --- Jalur 2: ensemble LBPH per-region (skoring relatif) ---
        acc = {}   # nama -> list[(z, bobot)] dari tiap region
        for r, w in weights.items():
            if w <= 0.01 or not ready.get(r):
                continue
            try:
                col = cv2.face.StandardCollector_create()
                lbph[r].predict_collect(crop_region(canon_gray, r), col)
                results = col.getResults()
            except Exception:
                continue
            if not results:
                continue

            # predict_collect memberi jarak ke SETIAP template. Kita ambil jarak
            # terbaik per-ORANG (bukan per-template), karena satu orang bisa punya
            # puluhan template hasil augmentasi.
            per_person = {}
            for lbl, dist in results:
                nm = id_to_name.get(int(lbl))
                if nm is None:
                    continue
                if dist < per_person.get(nm, 1e18):
                    per_person[nm] = float(dist)
            if not per_person:
                continue

            for nm, d in per_person.items():
                if d > LBPH_HARD_MAX:
                    continue
                z = _standout(d, [v for k, v in per_person.items() if k != nm])
                acc.setdefault(nm, []).append((min(1.0, z / Z_REF), w))

            if detail:
                bn = min(per_person, key=per_person.get)
                per_region_dbg[r] = {
                    "nama": bn, "jarak": round(per_person[bn], 1),
                    "z": round(_standout(per_person[bn],
                               [v for k, v in per_person.items() if k != bn]), 2),
                    "bobot": w,
                }

        if acc:
            for nm, contribs in acc.items():
                lb = _fuse_regions(contribs)
                # Kalau SFace aktif, LBPH jadi pendukung (0.35) bukan penentu.
                scores[nm] = (0.65 * scores.get(nm, 0.0) + 0.35 * lb) if used_sface \
                    else max(scores.get(nm, 0.0), lb)

        if not scores:
            return {"name": None, "score": 0.0, "occlusion": occ,
                    "mode": "sface" if used_sface else "lbph", "detail": per_region_dbg}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0

        ok = (best >= ACCEPT_SIM) and ((best - second) >= ACCEPT_MARGIN or len(ranked) == 1)
        return {
            "name": best_name if ok else None,
            "score": round(float(best), 3),
            "runner_up": round(float(second), 3),
            "occlusion": occ,
            "mode": "sface+lbph" if used_sface else "lbph",
            "detail": per_region_dbg,
        }

    def recognize_frame(self, frame_bgr: np.ndarray, det: dict, detail: bool = False) -> dict:
        canon = self.align(frame_bgr, det)
        embed = self._embed(frame_bgr, det) if self.sface_available else None
        out = self.recognize(canon, embed, detail=detail)
        out["canon"] = canon
        return out

    def status(self) -> dict:
        with self._model_lock:
            st = dict(self.stats)
        st.update({
            "yunet": self.yunet_available, "yunet_path": YUNET_PATH,
            "sface": self.sface_available, "sface_path": SFACE_PATH,
            "mode": ENGINE_MODE,
            "regions": list(REGIONS.keys()),
            "lbph_hard_max": LBPH_HARD_MAX,
            "accept_sim": ACCEPT_SIM, "accept_margin": ACCEPT_MARGIN,
            "sface_cosine": SFACE_COSINE,
            "augment": AUG_ENABLED, "aug_per_photo": AUG_PER_PHOTO,
        })
        return st

    # ---------- SELF TEST ----------
    def selftest(self, rows) -> dict:
        """Ambil tiap foto terdaftar, tempel masker/topi/kerudung SINTETIS,
        lalu cek apakah masih dikenali dengan benar. Ini cara cepat mengukur
        apakah tuning ambang Anda sudah pas SEBELUM turun ke lapangan.
        """
        cases = {"bersih": lambda a: a,
                 "masker": lambda a: _apply_mask(a, 205),
                 "masker_gelap": lambda a: _apply_mask(a, 80, low=True),
                 "topi": lambda a: _apply_cap(a, 0.28, 45),
                 "kerudung": lambda a: _apply_hijab(a, False),
                 "kerudung+masker": lambda a: _apply_mask(_apply_hijab(a, False), 205),
                 "topi+masker": lambda a: _apply_mask(_apply_cap(a, 0.26, 55), 205)}
        res = {k: {"benar": 0, "salah": 0, "tidak_dikenal": 0} for k in cases}

        for p_id, name, img_path in rows:
            if not img_path or not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            dets = self.detect(img)
            try:
                canon = self.align(img, dets[0]) if dets else normalize_gray(img)
            except Exception:
                continue
            # blur ringan supaya uji tidak terlalu optimis dibanding stream asli
            canon = _noise(_blurify(canon, 0.6), 5, int(p_id))
            for k, fn in cases.items():
                probe = fn(canon)
                e = self._embed(None, None, canon_gray=probe) if self.sface_available else None
                r = self.recognize(probe, e)
                if r["name"] is None:
                    res[k]["tidak_dikenal"] += 1
                elif r["name"] == name:
                    res[k]["benar"] += 1
                else:
                    res[k]["salah"] += 1

        for k, v in res.items():
            tot = v["benar"] + v["salah"] + v["tidak_dikenal"]
            v["akurasi"] = round(100.0 * v["benar"] / tot, 1) if tot else 0.0
        return res


# =====================================================================
# VOTING TEMPORAL PER-TRACK
# =====================================================================

class TrackVoter:
    """Keputusan nama diambil dari mayoritas beberapa siklus, bukan 1 frame.

    Saat wajah tertutup, skor 1 frame bisa meleset. Dengan jendela 5 siklus dan
    syarat 3 suara sepakat, salah-sebut-nama turun drastis sementara latensi
    hanya bertambah ~2 siklus deteksi.
    """

    def __init__(self, window: int = VOTE_WINDOW, min_agree: int = VOTE_MIN_AGREE):
        self.window = window
        self.min_agree = min_agree
        self.votes = {}      # track_id -> list[str|None]
        self.decided = {}    # track_id -> str|None

    def push(self, track_id, name: Optional[str]) -> Optional[str]:
        buf = self.votes.setdefault(track_id, [])
        buf.append(name)
        if len(buf) > self.window:
            buf.pop(0)
        tally = {}
        for n in buf:
            if n:
                tally[n] = tally.get(n, 0) + 1
        if tally:
            best, cnt = max(tally.items(), key=lambda kv: kv[1])
            if cnt >= self.min_agree:
                self.decided[track_id] = best
                return best
        return self.decided.get(track_id)

    def get(self, track_id) -> Optional[str]:
        return self.decided.get(track_id)

    def purge(self, alive_ids: set):
        for tid in list(self.votes.keys()):
            if tid not in alive_ids:
                self.votes.pop(tid, None)
                self.decided.pop(tid, None)


# Instance global dipakai server.py
engine = FaceEngine()
