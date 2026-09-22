"""
AI Surveillance VMS Server - Multi-Camera + NX Optic Integration
==================================================================
Fitur:
1. Multi-camera grid (10+ kamera, dynamic add/remove via dashboard)
2. Per-camera AI toggle (face recognition, smoking, fire, behavior)
3. Integrasi penuh NX Optic: Generic Events, Bookmarks, Analytics Metadata
4. Webhook listener dari NX
"""

import cv2
import threading
import time
import os
import sqlite3
import asyncio
import numpy as np
import base64
import requests
import json
import uuid
import socket
import collections
import subprocess
import shutil
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime
from typing import Optional
import enpidix_fleet                        # setelah app = FastAPI(...), ~baris 139
import onvif_ptz                            # 📡 Auto-add kamera via ONVIF + kontrol PTZ multi-brand

# --- 🍏🚀 DETEKSI HARDWARE OTOMATIS ---
# Sebelumnya blok ini cuma menangani CPU lemah (mis. MacBook Air 2017 dual-core,
# tanpa GPU) dengan membatasi thread OpenCV/Torch supaya beberapa kamera tidak
# rebutan CPU. Sekarang ditambah deteksi platform NVIDIA Jetson (Orin NX/Nano/AGX)
# -- di Jetson, YOLO idealnya jalan di GPU (CUDA)/DLA lewat engine TensorRT, BUKAN
# di-throttle seperti CPU lemah, karena GPU Ampere-nya yang menangani beban berat,
# bukan CPU ARM-nya. Menganggap Jetson sebagai "low power" justru salah kaprah dan
# bikin YOLO_EVERY_N_FRAMES/YOLO_IMG_SIZE turun jauh lebih rendah dari yang GPU-nya
# sanggup, padahal Orin NX (100-157 TOPS) jauh lebih kuat dari CPU 6-core-nya sendiri.
CPU_CORES = os.cpu_count() or 4


def _detect_jetson() -> Optional[str]:
    """Kembalikan nama model board (mis. 'NVIDIA Orin NX Developer Kit') kalau server
    berjalan di Jetson, atau None kalau bukan (PC/Mac/cloud VM biasa). Dicek lewat
    device-tree model (paling akurat, tersedia dari JetPack manapun) lalu fallback
    ke penanda /etc/nv_tegra_release yang cuma ada di L4T/JetPack."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip("\x00").strip()
            if model:
                return model
    except Exception:
        pass
    if os.path.exists("/etc/nv_tegra_release"):
        return "NVIDIA Jetson (model spesifik tidak terbaca, tapi L4T terdeteksi)"
    return None


JETSON_MODEL = _detect_jetson()
IS_JETSON = JETSON_MODEL is not None
CUDA_AVAILABLE = False  # diisi ulang di bawah setelah torch berhasil di-import

LOW_POWER_MODE = os.environ.get("LOW_POWER_MODE", "auto").lower()
if LOW_POWER_MODE == "auto":
    # Jetson TIDAK dianggap low-power meski CPU ARM-nya cuma 6-12 core -- beban berat
    # (YOLO) lari di GPU/DLA, bukan CPU, jadi heuristik skip-frame agresif ala CPU
    # lemah justru kontraproduktif dan menyia-nyiakan kapasitas GPU yang ada.
    LOW_POWER_MODE = (CPU_CORES <= 4) and not IS_JETSON
else:
    LOW_POWER_MODE = LOW_POWER_MODE in ("1", "true", "yes", "on")

_cv_threads = max(1, CPU_CORES - 1)
cv2.setNumThreads(_cv_threads)
os.environ.setdefault("OMP_NUM_THREADS", str(_cv_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_cv_threads))
try:
    import torch
    torch.set_num_threads(_cv_threads)
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    pass

if IS_JETSON:
    print(f"🚀 Board Jetson terdeteksi: {JETSON_MODEL}")
    print(f"   CUDA terbaca oleh torch: {'YA ✅' if CUDA_AVAILABLE else 'TIDAK ❌'}")
    if not CUDA_AVAILABLE:
        print("   ⚠️  Jetson terdeteksi tapi CUDA tidak terbaca -- biasanya artinya torch "
              "yang terpasang adalah wheel PyPI biasa (CPU-only), bukan build khusus "
              "Jetson dari JetPack/l4t. YOLO akan fallback ke CPU (lambat). Lihat "
              "scripts/setup_jetson_orin_nx.sh untuk pasang torch/torchvision yang benar.")

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request, BackgroundTasks, Depends, Body
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import secrets

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil tidak terpasang. Jalankan: pip install psutil --break-system-packages "
          "agar endpoint /api/stats/system bisa menampilkan data CPU/RAM asli.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# GPUtil bergantung pada `nvidia-smi`, yang TIDAK ADA di Jetson (GPU-nya terintegrasi
# di SoC, dimonitor lewat sysfs/tegrastats, bukan lewat driver desktop biasa). Di Jetson,
# sumber statistik GPU/suhu/power yang benar adalah jetson-stats (jtop) -- pustaka resmi
# komunitas yang dipakai NVIDIA sendiri di dokumentasinya.
JTOP_AVAILABLE = False
if IS_JETSON:
    try:
        from jtop import jtop
        JTOP_AVAILABLE = True
    except ImportError:
        print("⚠️  Jetson terdeteksi tapi jetson-stats (jtop) belum terpasang -- GPU load/suhu/"
              "power di /api/stats/system tidak akan terisi. Pasang dengan: "
              "sudo pip3 install -U jetson-stats && sudo systemctl restart jtop.service, "
              "lalu restart server ini. Lihat scripts/setup_jetson_orin_nx.sh.")

# =========================================================
# ⚙️  KONFIGURASI UTAMA
# =========================================================
FACES_DIR = "registered_faces"
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

# --- 🎬 REKAMAN (NVR) ---
# Direkam LANGSUNG dari sumber RTSP pakai ffmpeg (subprocess terpisah per kamera),
# BUKAN dari frame hasil pipeline AI. Ini penting supaya rekaman mulus & timing-nya
# presisi mengikuti sumber asli — tidak ikut terpotong-potong oleh frame-skip/pacing
# TARGET_STREAM_FPS yang dipakai jalur live-AI (lihat catatan optimasi CPU sebelumnya).
#
# Lokasi penyimpanan rekaman SEKARANG DINAMIS (bisa pilih drive/folder lewat dashboard,
# lihat StorageManager di bawah) — RECORDINGS_DIR_DEFAULT cuma dipakai SEKALI sebagai
# folder awal ("primary") kalau admin belum pernah atur lokasi penyimpanan sama sekali
# (instalasi baru / upgrade dari versi sebelum fitur multi-disk ini ada).
RECORDINGS_DIR_DEFAULT = os.path.abspath(os.environ.get("RECORDINGS_DIR", "recordings"))
os.makedirs(RECORDINGS_DIR_DEFAULT, exist_ok=True)

# --- 📸 HASIL CAPTURE VISUAL (snapshot alert: wajah/asap-api/rokok/jatuh/counting) ---
# Sebelumnya snapshot alert cuma disimpan di memori (worker.latest_alerts, maks 50,
# hilang saat server restart). Sekarang setiap snapshot JUGA ditulis ke disk sebagai
# .jpg + sidecar .json (metadata: type/name/waktu) supaya bisa di-browse & "diputar
# ulang" (playback galeri) kapan saja lewat endpoint /api/cameras/{id}/captures,
# persis seperti rekaman video tapi untuk hasil capture gambar.
CAPTURES_DIR_DEFAULT = os.path.abspath(os.environ.get("CAPTURES_DIR", "captures"))
os.makedirs(CAPTURES_DIR_DEFAULT, exist_ok=True)
CAPTURE_RETENTION_DAYS_DEFAULT = int(os.environ.get("CAPTURE_RETENTION_DAYS", "30"))  # 0 = simpan selamanya
CAPTURE_FILENAME_RE = re.compile(r"^\d{8}_\d{6}_[a-z_]+_[0-9a-f]{6}\.jpg$")

# --- 💾 FAILOVER PENYIMPANAN ---
# Kalau lokasi penyimpanan aktif bermasalah (disk lepas/tidak ke-mount, disk penuh,
# gagal tulis I/O error), sistem otomatis pindah ke lokasi cadangan berikutnya sesuai
# urutan prioritas yang diatur admin — rekaman tidak berhenti total gara-gara 1 disk mati.
MIN_FREE_SPACE_GB = float(os.environ.get("MIN_FREE_SPACE_GB", "2"))       # di bawah ini dianggap 'penuh/bermasalah'
STORAGE_HEALTH_CHECK_INTERVAL_SEC = int(os.environ.get("STORAGE_HEALTH_CHECK_INTERVAL_SEC", "30"))
STORAGE_RECOVERY_STABLE_CHECKS = 3   # primary harus sehat N kali cek berturut baru dianggap "pulih" (anti flip-flop)

FFMPEG_BIN = shutil.which("ffmpeg")
if not FFMPEG_BIN:
    print("⚠️  ffmpeg TIDAK ditemukan di PATH server ini — fitur rekaman tidak akan bisa "
          "dijalankan sampai ffmpeg terpasang. Live streaming & AI analytics TIDAK terpengaruh "
          "(itu jalur terpisah, tidak butuh ffmpeg). Install: Windows (winget install ffmpeg / "
          "https://www.gyan.dev/ffmpeg/builds/), macOS (brew install ffmpeg), Ubuntu/Debian "
          "(sudo apt install ffmpeg).")

RECORD_SEGMENT_MIN_DEFAULT = int(os.environ.get("RECORD_SEGMENT_MIN", "15"))   # menit per file
RECORD_RETENTION_DAYS_DEFAULT = int(os.environ.get("RECORD_RETENTION_DAYS", "14"))  # 0 = simpan selamanya
RECORD_CRF_DEFAULT = int(os.environ.get("RECORD_CRF", "23"))  # 0-51, makin besar makin kompres/kecil filenya
RECORD_RESOLUTIONS = {  # preset resolusi output rekaman (kompres ukuran file)
    "original": None,
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
}
# Nama file HARUS cocok pola ini (dibuat sendiri oleh ffmpeg lewat -strftime) — dipakai
# untuk validasi ketat di endpoint download/hapus supaya tidak bisa path traversal.
RECORD_FILENAME_RE = re.compile(r"^\d{8}_\d{6}\.mp4$")

app = FastAPI(title="AI Surveillance VMS — Multi-Camera NX Integrated")
app.mount("/faces", StaticFiles(directory=FACES_DIR), name="faces")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# =========================================================
# 🔐 AUTENTIKASI DASHBOARD / API
# =========================================================
# Kredensial diambil dari environment variable, JANGAN hardcode.
# Wajib diset sebelum menjalankan server:
#   export VMS_USERNAME="admin"
#   export VMS_PASSWORD="ganti-dengan-password-kuat"
#   export VMS_SESSION_SECRET="$(openssl rand -hex 32)"
VMS_USERNAME       = os.environ.get("VMS_USERNAME", "enpidix")
VMS_PASSWORD       = os.environ.get("VMS_PASSWORD", "ytex2026")
VMS_SESSION_SECRET = os.environ.get("VMS_SESSION_SECRET", "$(openssl rand -hex 32)")

if not VMS_USERNAME or not VMS_PASSWORD:
    raise RuntimeError(
        "VMS_USERNAME dan VMS_PASSWORD wajib diset sebagai environment variable "
        "sebelum menjalankan server ini (jangan pakai default/kosong)."
    )
if not VMS_SESSION_SECRET:
    raise RuntimeError(
        "VMS_SESSION_SECRET wajib diset (contoh: export VMS_SESSION_SECRET=$(openssl rand -hex 32))."
    )

_basic_security = HTTPBasic(auto_error=False)

# Path yang TIDAK memerlukan login (halaman login itu sendiri, aset statis publik, health check)
_PUBLIC_PATHS = {"/login", "/health", "/manifest.json", "/sw.js"}
_PUBLIC_PREFIXES = ("/faces/", "/static/")

# Webhook dari NX Optic tidak melalui browser, jadi diamankan pakai shared-secret
# terpisah lewat header, bukan session/basic auth.
NX_WEBHOOK_SECRET = os.environ.get("NX_WEBHOOK_SECRET", "")


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Semua request harus terautentikasi, kecuali:
    - /login, /health, /faces/*, /static/* (aset publik)
    - /nx/webhook (diamankan sendiri via header X-Webhook-Secret, lihat handler-nya)
    Autentikasi diterima lewat salah satu dari:
    - Session cookie (setelah login lewat form /login) -> untuk browser/dashboard
    - HTTP Basic Auth header -> untuk akses API langsung / script
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES) or path == "/nx/webhook":
            return await call_next(request)

        # 1) Cek session (login via dashboard)
        if request.session.get("authenticated"):
            return await call_next(request)

        # 2) Cek HTTP Basic Auth (untuk akses API/script langsung)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            try:
                import base64 as _b64
                decoded = _b64.b64decode(auth_header[6:]).decode("utf-8")
                user, _, pw = decoded.partition(":")
                if secrets.compare_digest(user, VMS_USERNAME) and secrets.compare_digest(pw, VMS_PASSWORD):
                    return await call_next(request)
            except Exception:
                pass

        # Browser (request HTML) -> redirect ke halaman login
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return RedirectResponse(url="/login", status_code=303)

        # Non-browser (API call) -> 401 dengan WWW-Authenticate agar bisa prompt Basic Auth
        return JSONResponse(
            status_code=401,
            content={"status": "error", "detail": "Unauthorized. Silakan login atau sertakan Basic Auth."},
            headers={"WWW-Authenticate": "Basic"},
        )


# CORS: dibutuhkan supaya app mobile (React Native) & client web lain yang connect
# lewat Tailscale (IP 100.x.x.x atau MagicDNS *.ts.net) bisa memanggil API ini.
# React Native fetch tidak selalu terikat aturan CORS browser, tapi WebView/tooling
# lain di dalamnya bisa. allow_origins "*" aman di sini karena akses tetap dikunci
# oleh AuthMiddleware (session/basic auth) di atas — CORS hanya soal origin browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)

# PENTING: SessionMiddleware didaftarkan PALING TERAKHIR.
# Di Starlette, middleware yang didaftarkan terakhir = paling luar = jalan PALING DULU.
# Jadi SessionMiddleware harus jalan sebelum AuthMiddleware, agar request.session
# sudah tersedia saat AuthMiddleware memeriksanya. Kalau urutan dibalik, akan muncul
# error: 'SessionMiddleware must be installed to access request.session'.
app.add_middleware(
    SessionMiddleware,
    secret_key=VMS_SESSION_SECRET,
    session_cookie="vms_session",
    same_site="lax",
    https_only=os.environ.get("VMS_COOKIE_HTTPS_ONLY", "false").lower() == "true",
    max_age=60 * 60 * 12,  # 12 jam
)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse(url="/", status_code=303)
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
      <meta charset="UTF-8">
      <title>Masuk — ENPIDIX VMS</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="manifest" href="/manifest.json">
      <meta name="theme-color" content="#0e1114">
      <link rel="apple-touch-icon" href="/static/icons/icon-192.png">
      <link rel="icon" type="image/png" href="data:image/png;base64,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">
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
      <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{background:#0e1114;color:#e6eaee;font:400 14px/1.45 'IBM Plex Sans',system-ui,sans-serif;
             height:100vh;display:flex;overflow:hidden}
        .art{flex:1;background:#0b0e11;position:relative;overflow:hidden}
        .art .grid{position:absolute;inset:0;background:repeating-linear-gradient(135deg,#12171b 0 14px,#0d1114 14px 28px)}
        .art .mark{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:22px}
        .art .logo{width:min(300px,44%);height:auto;display:block}
        .art .sub{font:400 12px 'IBM Plex Mono',monospace;color:rgba(255,255,255,.3)}
        .panel{width:min(560px,100%);background:#12161a;display:flex;flex-direction:column;
               justify-content:center;padding:0 72px;gap:26px;border-left:1px solid rgba(255,255,255,.07)}
        .word{font:200 20px 'IBM Plex Mono',monospace;letter-spacing:.05em;color:#02CBE2;text-transform:uppercase}
        .word b{font-weight:600}
        .rule{height:1px;background:rgba(255,255,255,.08)}
        h1{font:600 22px 'IBM Plex Sans'}
        .lede{font-size:14px;color:rgba(255,255,255,.5)}
        form{display:flex;flex-direction:column;gap:16px}
        label{display:flex;flex-direction:column;gap:8px;font-size:13px;color:rgba(255,255,255,.6)}
        input{background:#0b0e11;border:1px solid rgba(255,255,255,.12);border-radius:6px;padding:14px 16px;
              color:#e6eaee;font:400 15px 'IBM Plex Sans';outline:none}
        input:focus{border-color:#4fc3ce}
        button{margin-top:6px;background:#4fc3ce;color:#06232a;border:none;border-radius:6px;padding:16px;
               font:600 15px 'IBM Plex Sans';cursor:pointer}
        button:hover{background:#7fd8e0}
        .foot{font:400 12px 'IBM Plex Mono',monospace;color:rgba(255,255,255,.3)}
        @media (max-width:860px){ .art{display:none} .panel{width:100%;padding:0 32px;border-left:none} }
      </style>
    </head>
    <body>
      <div class="art">
        <div class="grid"></div>
        <div class="mark">
          <img class="logo" src="data:image/png;base64,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" alt="ENPIDIX">
          <div class="sub">kelola duniamu tanpa ribet</div>
        </div>
      </div>
      <div class="panel">
        <div class="word"><b>Video Management System</b></div>
        <div class="rule"></div>
        <div>
          <h1>Masuk ke sistem</h1>
          <div class="lede" style="margin-top:6px">Gunakan akun operator yang diberikan admin.</div>
        </div>
        <form method="post" action="/login">
          <label>Nama pengguna<input type="text" name="username" required autofocus autocomplete="username"></label>
          <label>Kata sandi<input type="password" name="password" required autocomplete="current-password"></label>
          <button type="submit">Masuk</button>
        </form>
        <div class="foot">Akses tercatat di server. Hubungi admin bila kata sandi lupa.</div>
      </div>
    </body>
    </html>
    """


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if secrets.compare_digest(username, VMS_USERNAME) and secrets.compare_digest(password, VMS_PASSWORD):
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=303)
    return HTMLResponse(
        content="""
        <script>alert('Username atau password salah.'); window.location.href='/login';</script>
        """,
        status_code=401,
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hardware": JETSON_MODEL if IS_JETSON else "non-Jetson",
        "cuda": CUDA_AVAILABLE,
        "yolo_model": YOLO_MODEL_PATH,
        "yolo_device": "cuda:0" if CUDA_AVAILABLE else "cpu",
    }


# =========================================================
# 📱 PWA — Manifest & Service Worker
# =========================================================

APP_NAME = "AI Surveillance VMS"

@app.get("/manifest.json")
async def pwa_manifest():
    return JSONResponse({
        "name": APP_NAME,
        "short_name": "VMS",
        "description": "AI Surveillance VMS - Multi-Camera Dashboard",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "any",
        "background_color": "#0f172a",
        "theme_color": "#0f172a",
        "icons": [
            {"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    })


@app.get("/sw.js")
async def service_worker():
    # Service worker minimal: cache aset statis untuk mempercepat load & memungkinkan
    # ikon/app terpasang offline-capable. Data kamera/API tetap selalu fetch fresh
    # dari network (tidak di-cache) karena sifatnya real-time.
    js = """
const CACHE_NAME = "vms-pwa-v1";
const PRECACHE_URLS = [
  "/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  // Jangan cache API/dashboard/login: harus selalu real-time & tetap lewat auth.
  if (url.pathname.startsWith("/api/") || url.pathname === "/" || url.pathname === "/login") {
    return;
  }
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
"""
    return Response(content=js, media_type="application/javascript")

DB_FILE = "vms.db"

# --- 🧕🧢😷 MESIN WAJAH TAHAN OKLUSI ---------------------------------------
# Haar cascade frontal (di bawah) dipertahankan HANYA sebagai jaring pengaman.
# Semua jalur deteksi & pengenalan wajah sekarang lewat enpidix_face_engine,
# yang menangani wajah berkerudung / bertopi / bermasker. Alasan teknis lengkap
# ada di docstring modul tersebut.
from enpidix_face_engine import (            # noqa: E402
    engine as face_engine,
    TrackVoter as FaceTrackVoter,
    normalize_gray as _engine_normalize,
)

FACIAL_CLASSIFIER = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- 📐 NORMALISASI CROP WAJAH (penting untuk akurasi LBPH) ---
# LBPH membandingkan pola tekstur lokal. Kalau foto yang didaftarkan (mis. upload
# foto resolusi tinggi, wajah besar di frame) tidak disamakan skalanya dengan wajah
# yang ditangkap live dari kamera (resolusi rendah, wajah kecil di frame), pola
# teksturnya jadi beda skala -> confidence score selalu tinggi -> selalu "Tidak
# Dikenal" walau orangnya sudah terdaftar. Semua crop wajah (saat training MAUPUN
# saat prediksi live) WAJIB melewati fungsi ini dulu supaya skala & kontrasnya sama.
FACE_NORM_SIZE = (200, 200)


def _normalize_face_gray(face_gray: np.ndarray) -> np.ndarray:
    """Samakan skala + kontras crop wajah.

    DIUBAH: dulu memakai cv2.equalizeHist yang bersifat GLOBAL — satu bayangan
    gelap besar dari lidah topi atau lipatan kerudung menggeser seluruh histogram
    dan ikut menghapus tekstur di bagian wajah yang sebenarnya masih terlihat.
    Sekarang memakai denoise ringan + CLAHE (perataan kontras per-tile 8x8),
    sehingga area terang tetap detail walau bersebelahan dengan area gelap pekat.
    """
    return _engine_normalize(face_gray)


# --- 🔭 NX OPTIC / WITNESS KONFIGURASI (server-level, dipakai semua kamera) ---
NX_SERVER_IP   = os.environ.get("NX_SERVER_IP",   "192.168.1.14")
NX_PORT        = os.environ.get("NX_PORT",        "7001")
NX_USER        = os.environ.get("NX_USER",        "admin")
NX_PASSWORD    = os.environ.get("NX_PASSWORD",    "haikal0904")
NX_PLUGIN_ID   = os.environ.get("NX_PLUGIN_ID",   "")
NX_USE_HTTPS   = os.environ.get("NX_USE_HTTPS", "true").lower() != "false"
_nx_scheme     = "https" if NX_USE_HTTPS else "http"
NX_BASE_URL    = f"{_nx_scheme}://{NX_SERVER_IP}:{NX_PORT}"

NX_COOLDOWN_SEC = 10
RECOGNITION_CONFIDENCE_THRESHOLD = 70
# Default sekarang mempertimbangkan 3 kondisi, bukan cuma LOW_POWER_MODE:
#   1) CPU lemah tanpa GPU (mis. MacBook Air 2017)       -> paling hemat (skip banyak)
#   2) Jetson TAPI CUDA tidak terbaca (torch salah build) -> tetap hemat, YOLO jalan di CPU ARM
#   3) Jetson DENGAN CUDA aktif (Orin NX/Nano/AGX normal) -> paling agresif, GPU yang kerja
_JETSON_GPU_OK = IS_JETSON and CUDA_AVAILABLE
YOLO_EVERY_N_FRAMES = int(os.environ.get(
    "YOLO_EVERY_N_FRAMES",
    "20" if LOW_POWER_MODE else ("4" if _JETSON_GPU_OK else "8")
))
FACE_EVERY_N_FRAMES = int(os.environ.get(
    "FACE_EVERY_N_FRAMES",
    "10" if LOW_POWER_MODE else ("3" if _JETSON_GPU_OK else "3")
))
# Ukuran minimum wajah yang dicari (px) -> mengurangi jumlah window yang discan cascade.
FACE_MIN_SIZE = (48, 48)
# Faktor downscale khusus TAHAP DETEKSI wajah (bukan tahap pengenalan) -> Haar cascade
# jauh lebih cepat di gambar kecil. Kotak wajah yang ditemukan di-scale balik ke resolusi
# penuh sebelum dipakai untuk face recognition, jadi akurasi pengenalan tidak berkurang.
# Haar cascade jalan di CPU (bukan GPU) walau di Jetson, jadi tetap ikut aturan LOW_POWER_MODE.
FACE_DETECT_SCALE = float(os.environ.get("FACE_DETECT_SCALE", "0.6" if LOW_POWER_MODE else "1.0"))
# Resolusi input YOLO (imgsz). Ultralytics default 640. Di CPU lemah diturunkan ke 320
# (inferensi ~3-4x lebih cepat, trade-off akurasi kecil). Di Jetson dengan GPU aktif,
# GPU Ampere-nya (apalagi lewat TensorRT engine) sanggup 640 penuh dengan FPS layak,
# jadi tidak perlu dikorbankan seperti di CPU biasa.
YOLO_IMG_SIZE = int(os.environ.get(
    "YOLO_IMG_SIZE",
    "320" if LOW_POWER_MODE else ("640" if _JETSON_GPU_OK else "480")
))
# Target FPS stream yang dipatok stabil per kamera (capture loop akan membuang frame
# ekstra di atas target ini, bukan memaksakan proses semua frame mentah dari kamera).
TARGET_STREAM_FPS = float(os.environ.get(
    "TARGET_STREAM_FPS",
    "20" if _JETSON_GPU_OK else "15"
))

# --- 🔥 FIRE / SMOKE DETECTION ---
# CATATAN PENTING: YOLOv8n bawaan (COCO) TIDAK PUNYA class "fire"/"smoke" sama sekali —
# sebelumnya kode ini mencocokkan cls_name ke {"fire","smoke","backpack"} yang artinya
# deteksi api/asap SELALU GAGAL (dead code, class itu tidak pernah muncul dari model COCO)
# dan "backpack" jelas tidak relevan. Diganti total dengan engine berbasis analisis warna
# (HSV) + luas area minimum + konfirmasi beberapa siklus berturut-turut (lihat
# OverlayTracker & _detect_fire_smoke_regions) — tanpa perlu model tambahan, jalan di CPU
# manapun, dan jauh lebih relevan untuk kobaran api & asap yang sebenarnya.
FIRE_SENSITIVITY = int(os.environ.get("FIRE_SENSITIVITY", "5"))          # 1 (longgar) - 10 (ketat)
FIRE_MIN_AREA_PCT = float(os.environ.get("FIRE_MIN_AREA_PCT", "0.6"))    # % luas frame minimum
FIRE_CONFIRM_FRAMES = int(os.environ.get("FIRE_CONFIRM_FRAMES", "3"))   # siklus berturut sebelum alert
FIRE_EVERY_N_FRAMES = int(os.environ.get("FIRE_EVERY_N_FRAMES", "6" if LOW_POWER_MODE else "3"))

# --- 🚬 SMOKING DETECTION ---
# Sebelumnya: SETIAP ada "cell phone"/"cup"/"bottle" di mana pun dalam frame langsung
# dianggap "Potensi Merokok" — sangat tidak akurat (siapapun pegang HP/minum kopi
# ke-flag). Diganti dengan heuristik spasial: objek kecil itu HARUS berada di sekitar
# area kepala/tangan-ke-mulut seseorang (bagian atas bounding box orang), bukan di
# bagian mana pun frame, + dikonfirmasi beberapa siklus berturut-turut sebelum alert.
# Catatan: ini tetap heuristik proxy (COCO tidak punya class "rokok"). Untuk akurasi
# maksimal, sambungkan model custom terlatih khusus rokok lewat CUSTOM_SMOKING_MODEL_PATH.
# DIGANTI TOTAL. Catatan di atas menyebut heuristik proxy ini "tetap heuristik",
# padahal masalahnya lebih dalam: logikanya TERBALIK. Rokok lebarnya ~8 mm, jadi
# pada frame 480-640 px YOLOv8n tidak akan pernah mengklasifikasikannya sebagai
# cup/bottle/cell phone — perokok asli hampir selalu LOLOS. Sebaliknya orang
# menelepon dan orang minum kopi kena flag hampir setiap kali. Menaikkan
# SMOKING_CONFIRM_FRAMES pun tidak menolong: konfirmasi berulang justru
# MEMPERKUAT false positive, karena orang menelepon memegang HP-nya jauh lebih
# lama daripada perokok mengangkat tangan.
#
# Sekarang ketiga class itu dipakai sebagai PENYANGKAL (mengurangi skor), dan
# penilaian sesungguhnya dilakukan enpidix_smoking_engine lewat 4 bukti
# terpisah — terutama pola SIKLUS isapan, satu-satunya isyarat yang benar-benar
# memisahkan merokok dari menelepon.
from enpidix_smoking_engine import (           # noqa: E402
    SmokingEngine,
    CONFUSER_CLASSES as SMOKING_CONFUSER_CLASSES,
)
import enpidix_smoking_engine as smoking_cfg   # noqa: E402

SMOKING_PROXY_CLASSES = set(SMOKING_CONFUSER_CLASSES)   # warisan: dipakai endpoint lama
SMOKING_HEAD_ZONE_PCT = float(os.environ.get("SMOKING_HEAD_ZONE_PCT", "28"))
SMOKING_CONFIRM_FRAMES = int(os.environ.get("SMOKING_CONFIRM_FRAMES", "4"))  # warisan, tidak dipakai
CUSTOM_SMOKING_MODEL_PATH = os.environ.get("CUSTOM_SMOKING_MODEL_PATH", "")

# --- 🚨 FALL / BEHAVIOR DETECTION ---
BEHAVIOR_CONFIRM_FRAMES = int(os.environ.get("BEHAVIOR_CONFIRM_FRAMES", "3"))

# --- 🅿️ PARKING DETECTION (custom polygon per slot) ---
# Tiap slot parkir digambar admin sebagai poligon custom (bukan garis lurus tetap)
# lewat dashboard, disimpan per-kamera di tabel parking_slots. Status terisi/kosong
# ditentukan dari titik tengah box kendaraan (hasil YOLO, class di VEHICLE_CLASSES)
# yang jatuh di dalam poligon slot tsb, dengan confirm-streak multi-siklus (macam
# fire/smoking) supaya status tidak kedip-kedip akibat noise 1 siklus deteksi.
PARKING_CONFIRM_FRAMES = int(os.environ.get("PARKING_CONFIRM_FRAMES", "5"))

# --- 🚗🧍🐾 KLASIFIKASI KATEGORI UNTUK PEOPLE / VEHICLE COUNTING ---
# Dipakai supaya satu inferensi YOLO yang sama bisa dipakai untuk membedakan
# ORANG, KENDARAAN (dengan sub-tipe mobil/motor/bus/truk), dan HEWAN sekaligus,
# lalu tiap kategori dihitung terpisah saat melewati virtual line.
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

VEHICLE_LABELS_ID = {
    "car": "Mobil", "motorcycle": "Motor", "bus": "Bus", "truck": "Truk", "bicycle": "Sepeda",
}
COUNTING_LABELS_ID = {"person": "Orang", "animal": "Hewan", **VEHICLE_LABELS_ID}
# Warna kotak overlay per kategori (BGR, dipakai cv2.rectangle)
COUNTING_COLORS = {
    "person": (46, 204, 113), "animal": (255, 193, 7),
    "car": (255, 140, 0), "motorcycle": (255, 0, 200), "bus": (0, 140, 255), "truck": (0, 90, 200), "bicycle": (200, 200, 0),
}

# Ini membatasi thread CAPTURE/decode paralel (bukan concurrent GPU inference --
# inferensi YOLO sendiri sudah diserialize lewat yolo_lock di bawah, satu GPU cuma bisa
# mengerjakan satu batch inferensi ultralytics dalam satu waktu). Di Jetson dengan GPU
# aktif, angka lebih tinggi aman karena decode video (banyak dipakai lewat hardware
# NVDEC/OpenCV) tidak memperebutkan resource yang sama dengan inferensi GPU.
MAX_CONCURRENT_AI_THREADS = int(os.environ.get(
    "MAX_CONCURRENT_AI_THREADS",
    "2" if LOW_POWER_MODE else ("10" if _JETSON_GPU_OK else "6")
))

# --- 🔄 LEGACY ENV VARS (dari server.py single-camera lama) ---
# Dipakai HANYA untuk migrasi otomatis: kalau database kamera masih kosong
# dan RTSP_URL ini diset, kamera pertama akan dibuat otomatis dari sini.
LEGACY_RTSP_URL     = os.environ.get("RTSP_URL", "")
LEGACY_NX_CAMERA_ID = os.environ.get("NX_CAMERA_ID", "")
LEGACY_CAMERA_NAME  = os.environ.get("LEGACY_CAMERA_NAME", "Kamera Utama (Migrasi)")

# =========================================================
# 🧠 DATABASE
# =========================================================
db_lock = threading.Lock()


def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            nx_camera_id TEXT DEFAULT '',
            enabled INTEGER DEFAULT 1,
            ai_face_recognition INTEGER DEFAULT 0,
            ai_smoking_detection INTEGER DEFAULT 0,
            ai_fire_detection INTEGER DEFAULT 0,
            ai_behavior_detection INTEGER DEFAULT 0,
            ai_people_counting INTEGER DEFAULT 0,
            ai_vehicle_counting INTEGER DEFAULT 0,
            ai_animal_counting INTEGER DEFAULT 0,
            count_line_x1 REAL DEFAULT 0.1,
            count_line_y1 REAL DEFAULT 0.5,
            count_line_x2 REAL DEFAULT 0.9,
            count_line_y2 REAL DEFAULT 0.5,
            count_shape TEXT DEFAULT 'line',
            count_direction TEXT DEFAULT 'both',
            count_polygon TEXT DEFAULT '[]',
            count_line_style TEXT DEFAULT 'rope',
            count_line_thickness REAL DEFAULT 2,
            record_enabled INTEGER DEFAULT 0,
            record_resolution TEXT DEFAULT 'original',
            record_crf INTEGER DEFAULT 23,
            record_segment_min INTEGER DEFAULT 15,
            record_retention_days INTEGER DEFAULT 14,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS storage_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            priority INTEGER NOT NULL DEFAULT 0,
            label TEXT DEFAULT '',
            enabled INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS camera_counts (
            camera_id INTEGER PRIMARY KEY,
            counts_json TEXT DEFAULT '{}',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parking_slots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            label TEXT DEFAULT '',
            polygon TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- 🔧 Migrasi ringan: kalau DB lama sudah ada sebelum kolom counting
    # ditambahkan, CREATE TABLE IF NOT EXISTS di atas tidak akan menambah kolom
    # baru ke tabel `cameras` yang sudah ada. Jadi dicek manual & ditambah lewat
    # ALTER TABLE kalau belum ada, supaya upgrade tidak perlu hapus database lama.
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(cameras)").fetchall()}
    migrations = {
        "ai_people_counting": "ALTER TABLE cameras ADD COLUMN ai_people_counting INTEGER DEFAULT 0",
        "ai_vehicle_counting": "ALTER TABLE cameras ADD COLUMN ai_vehicle_counting INTEGER DEFAULT 0",
        "ai_animal_counting": "ALTER TABLE cameras ADD COLUMN ai_animal_counting INTEGER DEFAULT 0",
        "count_line_x1": "ALTER TABLE cameras ADD COLUMN count_line_x1 REAL DEFAULT 0.1",
        "count_line_y1": "ALTER TABLE cameras ADD COLUMN count_line_y1 REAL DEFAULT 0.5",
        "count_line_x2": "ALTER TABLE cameras ADD COLUMN count_line_x2 REAL DEFAULT 0.9",
        "count_line_y2": "ALTER TABLE cameras ADD COLUMN count_line_y2 REAL DEFAULT 0.5",
        "count_shape": "ALTER TABLE cameras ADD COLUMN count_shape TEXT DEFAULT 'line'",
        "count_direction": "ALTER TABLE cameras ADD COLUMN count_direction TEXT DEFAULT 'both'",
        "count_polygon": "ALTER TABLE cameras ADD COLUMN count_polygon TEXT DEFAULT '[]'",
        "count_line_style": "ALTER TABLE cameras ADD COLUMN count_line_style TEXT DEFAULT 'rope'",
        "count_line_thickness": "ALTER TABLE cameras ADD COLUMN count_line_thickness REAL DEFAULT 2",
        "record_enabled": "ALTER TABLE cameras ADD COLUMN record_enabled INTEGER DEFAULT 0",
        "record_resolution": "ALTER TABLE cameras ADD COLUMN record_resolution TEXT DEFAULT 'original'",
        "record_crf": f"ALTER TABLE cameras ADD COLUMN record_crf INTEGER DEFAULT {RECORD_CRF_DEFAULT}",
        "record_segment_min": f"ALTER TABLE cameras ADD COLUMN record_segment_min INTEGER DEFAULT {RECORD_SEGMENT_MIN_DEFAULT}",
        "record_retention_days": f"ALTER TABLE cameras ADD COLUMN record_retention_days INTEGER DEFAULT {RECORD_RETENTION_DAYS_DEFAULT}",
        "ai_parking_detection": "ALTER TABLE cameras ADD COLUMN ai_parking_detection INTEGER DEFAULT 0",
        # --- 📡 ONVIF + PTZ (opsional — kosong utk kamera RTSP manual biasa) ---
        "onvif_host": "ALTER TABLE cameras ADD COLUMN onvif_host TEXT DEFAULT ''",
        "onvif_port": "ALTER TABLE cameras ADD COLUMN onvif_port INTEGER DEFAULT 80",
        "onvif_username": "ALTER TABLE cameras ADD COLUMN onvif_username TEXT DEFAULT ''",
        "onvif_password": "ALTER TABLE cameras ADD COLUMN onvif_password TEXT DEFAULT ''",
        "onvif_profile_token": "ALTER TABLE cameras ADD COLUMN onvif_profile_token TEXT DEFAULT ''",
        "ptz_supported": "ALTER TABLE cameras ADD COLUMN ptz_supported INTEGER DEFAULT 0",
        "camera_brand": "ALTER TABLE cameras ADD COLUMN camera_brand TEXT DEFAULT ''",
    }
    for col, ddl in migrations.items():
        if col not in existing_cols:
            conn.execute(ddl)

    conn.commit()
    conn.close()


init_db()

recognizer_ready = False
id_to_name = {}
face_train_stats = {}


def _person_rows():
    with db_lock:
        conn = get_conn()
        rows = conn.execute("SELECT id, name, image_path FROM persons").fetchall()
        conn.close()
    return [(r["id"], r["name"], r["image_path"]) for r in rows]


def train_recognizer():
    """Latih ulang mesin wajah dari seluruh foto terdaftar.

    Perbedaan besar dibanding versi lama: tiap foto tidak lagi masuk sebagai SATU
    sampel wajah terbuka. Foto di-align dulu memakai titik mata, lalu diperbanyak
    otomatis jadi puluhan sampel yang meniru kondisi lapangan — versi BERMASKER,
    BERTOPI, dan BERKERUDUNG sintetis, plus variasi buram/berderau/miring/redup.
    Hasilnya orang yang mendaftar dengan satu foto wajah terbuka tetap dikenali
    saat datang memakai kerudung, topi, atau masker.

    Return: jumlah sampel latih (bukan jumlah foto) — dipakai dashboard.
    """
    global recognizer_ready, id_to_name, face_train_stats
    rows = _person_rows()
    stats = face_engine.train(rows)
    face_train_stats = stats
    id_to_name = dict(face_engine.id_to_name)
    recognizer_ready = stats["samples"] > 0
    print(f"[FaceEngine] {stats['persons']} orang / {stats['photos']} foto "
          f"-> {stats['samples']} sampel, {stats['embeddings']} embedding "
          f"({stats['train_seconds']}s)")
    return stats["samples"]


train_recognizer()

# --- 🧠 MODEL YOLO: pilih device & format model otomatis ---
# - Ada GPU CUDA? (termasuk GPU Ampere Jetson Orin NX/Nano/AGX) -> pakai GPU, bukan CPU.
# - Ada file .engine hasil export TensorRT KHUSUS board ini (lihat
#   scripts/export_tensorrt_jetson.sh) -> load itu. Jauh lebih cepat & hemat memori
#   daripada .pt biasa karena sudah dikompilasi + dikuantisasi (FP16) untuk GPU board ini.
#   PENTING: file .engine TIDAK portable antar board / versi JetPack-TensorRT -- harus
#   di-export ULANG langsung di board target, tidak bisa disalin dari board lain / dari PC.
# - Belum ada .engine -> fallback ke .pt apa adanya (tetap jalan di GPU kalau CUDA_AVAILABLE,
#   cuma belum secepat versi TensorRT).
YOLO_DEVICE = 0 if CUDA_AVAILABLE else "cpu"
_yolo_engine_candidate = os.environ.get("YOLO_ENGINE_PATH", "yolov8n.engine")
YOLO_MODEL_PATH_DEFAULT = _yolo_engine_candidate if (CUDA_AVAILABLE and os.path.exists(_yolo_engine_candidate)) else "yolov8n.pt"
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", YOLO_MODEL_PATH_DEFAULT)

# YOLO model di-load sekali, dipakai semua camera worker (thread-safe untuk inference)
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_lock = threading.Lock()  # ultralytics predict tidak 100% thread-safe, kita serialize

print(f"🧠 YOLO model dimuat: {YOLO_MODEL_PATH}  |  device: {'GPU cuda:0' if CUDA_AVAILABLE else 'CPU'}"
      + (f"  |  Jetson: {JETSON_MODEL}" if IS_JETSON else ""))
if IS_JETSON and CUDA_AVAILABLE and not YOLO_MODEL_PATH.endswith(".engine"):
    print("   💡 Tip performa: export model ke TensorRT engine supaya inferensi jauh lebih "
          "cepat & hemat memori di board ini -> jalankan scripts/export_tensorrt_jetson.sh "
          "sekali di board Jetson ini, lalu restart server (otomatis kepakai kalau file "
          "yolov8n.engine ditemukan).")

if not os.path.exists(smoking_cfg.POSE_MODEL_PATH):
    print(f"ℹ️  Model pose '{smoking_cfg.POSE_MODEL_PATH}' belum ada. Deteksi merokok "
          f"tetap jalan, tapi bukti GESTUR memakai perkiraan gerakan yang jauh lebih "
          f"lemah — manfaat utamanya jadi sebatas menghapus false positive HP/gelas. "
          f"Unduh sekali: yolo export tidak perlu, cukup "
          f"`python -c \"from ultralytics import YOLO; YOLO('yolov8n-pose.pt')\"` "
          f"lalu pindahkan file .pt ke folder models/.")

if CUSTOM_SMOKING_MODEL_PATH:
    print(f"⚠️  CUSTOM_SMOKING_MODEL_PATH diset ke '{CUSTOM_SMOKING_MODEL_PATH}' tapi belum ada "
          f"kode yang memakainya — variabel ini baru DISEDIAKAN sebagai titik ekstensi utk model "
          f"deteksi rokok custom di masa depan, deteksi merokok saat ini tetap pakai heuristik "
          f"spasial (objek dekat zona kepala/tangan-ke-mulut). Aman diabaikan / dikosongkan.")

# =========================================================
# 📡 NX OPTIC INTEGRATION LAYER
# =========================================================

_nx_token_cache: dict = {"token": None, "expires_at": 0.0}
_nx_token_lock  = threading.Lock()
nx_cooldown_tracker: dict = {}
nx_status = {
    "reachable": None,
    "last_error": None,
    "last_changed_at": None,   # timestamp (epoch) saat status terakhir berubah
    "last_checked_at": None,   # timestamp polling terakhir
    "since": None,             # ISO string kapan status saat ini mulai berlaku
}
nx_status_lock = threading.Lock()
NX_POLL_INTERVAL_SEC = int(os.environ.get("NX_POLL_INTERVAL_SEC", "3"))


def _nx_get_token() -> Optional[str]:
    """Login ke NX Optic v6, ambil Bearer Token. Auto fallback HTTPS->HTTP."""
    global NX_BASE_URL, _nx_scheme

    with _nx_token_lock:
        if _nx_token_cache["token"] and time.time() < _nx_token_cache["expires_at"]:
            return _nx_token_cache["token"]

        if not NX_SERVER_IP or not NX_USER or not NX_PASSWORD:
            nx_status["reachable"] = False
            nx_status["last_error"] = "NX_SERVER_IP/NX_USER/NX_PASSWORD belum diset"
            return None

        login_payload = {"username": NX_USER, "password": NX_PASSWORD, "setCookie": False}
        schemes_to_try = ["https", "http"] if NX_USE_HTTPS else ["http"]

        for scheme in schemes_to_try:
            base = f"{scheme}://{NX_SERVER_IP}:{NX_PORT}"
            try:
                resp = requests.post(
                    f"{base}/rest/v3/login/sessions",
                    json=login_payload, timeout=8, verify=False,
                )
                if resp.ok:
                    data = resp.json()
                    token = (
                        data.get("token") or data.get("authToken")
                        or data.get("access_token")
                        or (data.get("reply", {}) or {}).get("token")
                    )
                    if token:
                        _nx_token_cache["token"] = token
                        _nx_token_cache["expires_at"] = time.time() + 3000
                        NX_BASE_URL = base
                        _nx_scheme = scheme
                        nx_status["reachable"] = True
                        nx_status["last_error"] = None
                        print(f"[NX Auth ✅] Token diperoleh via {scheme.upper()}")
                        return token
                    nx_status["last_error"] = f"Token tidak ditemukan: {list(data.keys())}"
                else:
                    nx_status["last_error"] = f"Login gagal HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.exceptions.SSLError as e:
                nx_status["last_error"] = f"SSL Error: {e}"
                continue
            except requests.exceptions.ConnectionError as e:
                nx_status["reachable"] = False
                nx_status["last_error"] = f"Tidak bisa konek ke {base}: {e}"
                continue
            except Exception as e:
                nx_status["last_error"] = str(e)

        nx_status["reachable"] = False
        print("[NX Auth ❌] Semua percobaan login gagal.")
        return None


def _nx_headers(with_auth: bool = True) -> dict:
    headers = {"Content-Type": "application/json"}
    if with_auth:
        token = _nx_get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def kirim_event_ke_nx(caption: str, description: str = "", nx_camera_id: str = ""):
    """
    Kirim Generic Event ke NX Optic/Witness.
    Endpoint resmi (terverifikasi via testing langsung & masuk ke Event Log NX):
    POST /api/createEvent — bukan /rest/v3/events/generic (endpoint itu tidak ada).
    Device dirujuk via metadata.cameraRefs (array of camera UUID).
    """
    if not NX_SERVER_IP or not NX_USER or not NX_PASSWORD:
        return

    cooldown_key = f"{nx_camera_id}:{caption}"
    now = time.time()
    if cooldown_key in nx_cooldown_tracker and (now - nx_cooldown_tracker[cooldown_key]) < NX_COOLDOWN_SEC:
        return
    nx_cooldown_tracker[cooldown_key] = now

    def _worker():
        def _do_post():
            payload = {
                "source": "AI_Surveillance_VMS",
                "caption": caption,
                "description": description,
                "timestamp": int(time.time() * 1000),
                **({"metadata": {"cameraRefs": [nx_camera_id]}} if nx_camera_id else {}),
            }
            return requests.post(
                f"{NX_BASE_URL}/api/createEvent",
                json=payload, headers=_nx_headers(), timeout=5, verify=False,
            )
        try:
            resp = _do_post()
            if resp.status_code == 401:
                _nx_token_cache["token"] = None
                resp = _do_post()
            nx_status["reachable"] = resp.ok
            nx_status["last_error"] = None if resp.ok else f"HTTP {resp.status_code}: {resp.text[:200]}"
            print(f"[NX Event {'✅' if resp.ok else '❌'}] {caption} (cam={nx_camera_id}) -> {resp.status_code}")
        except Exception as e:
            nx_status["reachable"] = False
            nx_status["last_error"] = str(e)
            print(f"[NX Event ERR] {e}")

    threading.Thread(target=_worker, daemon=True).start()


def buat_bookmark_nx(name: str, description: str = "", nx_camera_id: str = "",
                      duration_ms: int = 5000, tags: Optional[list] = None):
    """
    Buat Bookmark di timeline kamera NX Optic/Witness.
    Endpoint resmi (per-device, terverifikasi): POST /rest/v3/devices/{deviceId}/bookmarks
    — bukan /rest/v3/bookmarks global (endpoint itu tidak ada).
    """
    if not NX_SERVER_IP or not NX_USER or not NX_PASSWORD or not nx_camera_id:
        return

    def _worker():
        def _do_post():
            payload = {
                "name": name, "description": description,
                "startTimeMs": int(time.time() * 1000),
                "durationMs": duration_ms,
                "tags": tags or ["AI_Alert"],
            }
            return requests.post(
                f"{NX_BASE_URL}/rest/v3/devices/{nx_camera_id}/bookmarks",
                json=payload, headers=_nx_headers(), timeout=5, verify=False,
            )
        try:
            resp = _do_post()
            if resp.status_code == 401:
                _nx_token_cache["token"] = None
                resp = _do_post()
            print(f"[NX Bookmark {'✅' if resp.ok else '❌'}] '{name}' (cam={nx_camera_id}) -> {resp.status_code}")
        except Exception as e:
            print(f"[NX Bookmark ERR] {e}")

    threading.Thread(target=_worker, daemon=True).start()


def kirim_metadata_analytics_nx(object_type_id: str, label: str, bbox: tuple,
                                 nx_camera_id: str, frame_width: int = 640, frame_height: int = 360):
    """Push object metadata ke NX Analytics Engine (butuh NX_PLUGIN_ID)."""
    if not NX_PLUGIN_ID or not NX_SERVER_IP or not NX_PASSWORD or not nx_camera_id:
        return

    def _worker():
        try:
            x, y, w, h = bbox
            bounding_box = {
                "x": round(x / frame_width, 4), "y": round(y / frame_height, 4),
                "width": round(w / frame_width, 4), "height": round(h / frame_height, 4),
            }
            payload = {
                "timestampUs": int(time.time() * 1_000_000),
                "durationUs": 500_000,
                "objects": [{
                    "typeId": object_type_id, "trackId": str(uuid.uuid4()),
                    "boundingBox": bounding_box,
                    "attributes": [{"name": "label", "value": label}],
                }],
            }
            url = f"{NX_BASE_URL}/rest/v3/analytics/engines/{NX_PLUGIN_ID}/deviceAgents/{nx_camera_id}/metadata"
            resp = requests.post(url, json=payload, headers=_nx_headers(), timeout=3, verify=False)
            if not resp.ok:
                print(f"[NX Analytics ❌] {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[NX Analytics ERR] {e}")

    threading.Thread(target=_worker, daemon=True).start()


def lapor_deteksi_nx(caption: str, description: str, nx_camera_id: str,
                      buat_bookmark: bool = True, tags: Optional[list] = None):
    kirim_event_ke_nx(caption, description, nx_camera_id)
    if buat_bookmark:
        buat_bookmark_nx(caption, description, nx_camera_id, tags=tags or ["AI_Alert"])

# =========================================================
# 🎯 SIMPLE CENTROID TRACKER (untuk people/vehicle counting)
# =========================================================
class OverlayTracker:
    """
    Tracker ringan khusus box overlay AI (face / smoking / fire / jatuh) — TERPISAH
    dari SimpleTracker (yang dipakai untuk people/vehicle/animal counting) supaya
    logika counting-crossing tidak ikut terganggu.

    Dua manfaat utama dibanding kirim box mentah tiap siklus deteksi:
    1. PERSISTENCE — box tetap "menempel" di posisi terakhir selama beberapa siklus
       walau siklus deteksi berikutnya kebetulan tidak menangkap object itu lagi
       (mis. wajah nengok sedikit, asap mengaburkan sesaat) -> tidak kedip-kedip,
       terasa mengikuti motion object bukan muncul-hilang tiap siklus YOLO/cascade.
    2. CONFIRM STREAK — box baru harus konsisten terdeteksi beberapa siklus berturut
       sebelum alert (notifikasi/NX event) betulan ditembak -> jauh mengurangi
       false-positive sekali kedip dari noise deteksi.

    Update per "family" (kelompok tipe) secara independen: track dari family lain
    (mis. "face" saat yang di-update cuma family "fire") sama sekali tidak disentuh.
    """

    def __init__(self, max_distance: int = 110, max_missed: int = 8):
        self.tracks: dict = {}   # tid -> {cx,cy,type,box,label,confirmed_label,missed,confirm}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, detections: list, families: set):
        """
        detections: list of dict {cx,cy,type,box,label} — HANYA berisi deteksi dari
        family yang sedang di-refresh siklus ini.
        families: set tipe yang sedang di-refresh siklus ini (wajib eksplisit, supaya
        track family lain tidak ikut kena "missed++" walau detections-nya kosong).
        Return: list dict {track_id,cx,cy,type,box,label,confirm,is_new}.
        """
        unmatched = list(range(len(detections)))

        for tid, tr in list(self.tracks.items()):
            if tr["type"] not in families:
                continue  # family lain -> jangan disentuh sama sekali
            best_idx, best_dist = None, self.max_distance
            for i in unmatched:
                d = detections[i]
                if d["type"] != tr["type"]:
                    continue
                dist = ((d["cx"] - tr["cx"]) ** 2 + (d["cy"] - tr["cy"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_idx = dist, i
            if best_idx is not None:
                d = detections[best_idx]
                tr["cx"], tr["cy"], tr["box"], tr["label"] = d["cx"], d["cy"], d["box"], d["label"]
                tr["missed"] = 0
                tr["confirm"] = tr.get("confirm", 0) + 1
                unmatched.remove(best_idx)
            else:
                tr["missed"] += 1
                tr["confirm"] = 0  # putus streak -> hitung ulang dari 0 kalau muncul lagi

        for i in unmatched:
            d = detections[i]
            if d["type"] not in families:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "cx": d["cx"], "cy": d["cy"], "type": d["type"], "box": d["box"],
                "label": d["label"], "missed": 0, "confirm": 1,
            }

        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if tr["type"] in families and tr["missed"] > self.max_missed:
                del self.tracks[tid]

        return [
            {"track_id": tid, "cx": tr["cx"], "cy": tr["cy"], "type": tr["type"],
             "box": tr["box"], "label": tr["label"], "confirm": tr.get("confirm", 0)}
            for tid, tr in self.tracks.items()
        ]

    def purge_types(self, types: set):
        """Hapus langsung semua track dengan type di dalam `types` — dipanggil tiap
        frame dengan daftar fitur AI yang SEDANG NONAKTIF, supaya box tidak nyangkut
        selamanya kalau admin mematikan sebuah fitur (mis. fire_detection dimatikan
        -> box api lama harus langsung hilang, bukan menunggu meluruh natural)."""
        if not types:
            return
        for tid in [tid for tid, tr in self.tracks.items() if tr["type"] in types]:
            del self.tracks[tid]

    def snapshot(self):
        """Semua track aktif saat ini (dipakai buat rebuild current_detections tiap
        frame walau AI tidak jalan siklus itu, supaya box tidak kedip kosong)."""
        return [
            {"track_id": tid, "cx": tr["cx"], "cy": tr["cy"], "type": tr["type"],
             "box": tr["box"], "label": tr["label"], "confirm": tr.get("confirm", 0)}
            for tid, tr in self.tracks.items()
        ]


class SimpleTracker:
    """
    Pelacak objek ringan berbasis centroid (bukan Kalman/DeepSORT) — cukup untuk
    kebutuhan counting garis virtual & overlay box yang "mengikuti motion" di CPU
    lemah. Tiap siklus deteksi, objek baru dicocokkan ke track existing lewat
    class yang sama + jarak centroid terdekat. Track yang tidak match beberapa
    siklus berturut-turut dianggap hilang dari frame & dihapus.
    """

    def __init__(self, max_distance: int = 90, max_missed: int = 12):
        self.tracks: dict = {}   # track_id -> {centroid, cls, box, missed, side, last_counted_at}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, detections: list):
        """detections: list of (cx, cy, cls_name, (x1,y1,x2,y2)). Return list track aktif."""
        unmatched = list(range(len(detections)))

        for tid, tr in list(self.tracks.items()):
            best_idx, best_dist = None, self.max_distance
            for i in unmatched:
                cx, cy, cls_name, box = detections[i]
                if cls_name != tr["cls"]:
                    continue
                dist = ((cx - tr["centroid"][0]) ** 2 + (cy - tr["centroid"][1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_idx = dist, i
            if best_idx is not None:
                cx, cy, cls_name, box = detections[best_idx]
                tr["centroid"] = (cx, cy)
                tr["box"] = box
                tr["missed"] = 0
                unmatched.remove(best_idx)
            else:
                tr["missed"] += 1

        for i in unmatched:
            cx, cy, cls_name, box = detections[i]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "centroid": (cx, cy), "cls": cls_name, "box": box,
                "missed": 0, "side": None, "last_counted_at": 0.0,
            }

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        return [(tid, tr["centroid"][0], tr["centroid"][1], tr["cls"], tr["box"]) for tid, tr in self.tracks.items()]


# =========================================================
# 🎥 MULTI-CAMERA WORKER MANAGER
# =========================================================
# Setiap kamera punya thread sendiri yang membaca frame terus-menerus.
# AI processing (YOLO/face) hanya jalan kalau toggle AI kamera tsb aktif.

class CameraWorker:
    """Mengelola satu kamera: capture thread + AI processing + state."""

    def __init__(self, cam_row: dict):
        self.id = cam_row["id"]
        self.name = cam_row["name"]
        self.rtsp_url = cam_row["rtsp_url"]
        self.nx_camera_id = cam_row["nx_camera_id"] or ""

        self.ai_settings = {
            "face_recognition": bool(cam_row["ai_face_recognition"]),
            "smoking_detection": bool(cam_row["ai_smoking_detection"]),
            "fire_detection": bool(cam_row["ai_fire_detection"]),
            "behavior_detection": bool(cam_row["ai_behavior_detection"]),
            "people_counting": bool(cam_row["ai_people_counting"]),
            "vehicle_counting": bool(cam_row["ai_vehicle_counting"]),
            "animal_counting": bool(cam_row["ai_animal_counting"]) if "ai_animal_counting" in cam_row.keys() else False,
            "parking_detection": bool(cam_row["ai_parking_detection"]) if "ai_parking_detection" in cam_row.keys() else False,
        }

        # --- 📏 Virtual line untuk counting (koordinat dinormalisasi 0-1) ---
        # Objek yang melewati garis ini (dari satu sisi ke sisi lain) dihitung
        # sebagai "masuk"/"keluar" tergantung arah lintasannya. Adjustable lewat
        # dashboard (drag titik ujung garis di atas preview kamera).
        self.count_line = (
            float(cam_row["count_line_x1"]), float(cam_row["count_line_y1"]),
            float(cam_row["count_line_x2"]), float(cam_row["count_line_y2"]),
        )
        # --- 🧭 Bentuk area counting: "line" (garis virtual) atau "polygon" (area
        # bebas yang bisa disesuaikan bentuknya). Untuk polygon, titik disimpan
        # sebagai list [[x,y], ...] dinormalisasi 0-1 relatif terhadap frame. ---
        self.count_shape = (cam_row["count_shape"] if "count_shape" in cam_row.keys() else None) or "line"
        # --- ↔️ Arah yang dihitung: "both" (masuk & keluar), "in" (masuk saja),
        # "out" (keluar saja). ---
        self.count_direction = (cam_row["count_direction"] if "count_direction" in cam_row.keys() else None) or "both"
        try:
            self.count_polygon = json.loads(cam_row["count_polygon"]) if ("count_polygon" in cam_row.keys() and cam_row["count_polygon"]) else []
        except Exception:
            self.count_polygon = []
        self.count_line_style = (cam_row["count_line_style"] if "count_line_style" in cam_row.keys() else None) or "rope"
        try:
            self.count_line_thickness = float(cam_row["count_line_thickness"]) if "count_line_thickness" in cam_row.keys() and cam_row["count_line_thickness"] else 2.0
        except Exception:
            self.count_line_thickness = 2.0

        self.tracker = SimpleTracker()          # khusus people/vehicle/animal counting
        self.overlay_tracker = OverlayTracker() # khusus box face/smoking/fire/jatuh (smooth-follow)
        # Voting nama wajah lintas frame per-track. Saat wajah terhalang, hasil
        # satu frame bisa meleset; menuntut beberapa siklus sepakat menekan
        # salah-sebut-nama sampai nyaris nol dengan tambahan latensi ~2 siklus.
        self.face_voter = FaceTrackVoter()
        # Deteksi merokok butuh RIWAYAT per-orang (siklus isapan terhitung dalam
        # puluhan detik), jadi state-nya hidup di worker kamera, bukan dihitung
        # ulang tiap frame seperti heuristik lama.
        # Folder terpisah dari galeri capture biasa: paket bukti berbentuk
        # DIREKTORI (berisi pre-roll, post-roll, crop, JSON), bukan file .jpg
        # tunggal, jadi tidak dicampur dengan snapshot alert lama.
        self.smoking_engine = SmokingEngine(
            self.id, self.name, os.path.join(CAPTURES_DIR_DEFAULT, "bukti_merokok"))
        self._last_tracked_objects: list = []   # cache hasil tracker counting terakhir (persist antar frame)
        self.counts: dict = self._load_counts()  # {"person": {"in":0,"out":0}, "car": {...}, ...}

        # --- 🅿️ Parking slots (poligon custom per slot, digambar admin lewat dashboard) ---
        self.parking_slots: list = self._load_parking_slots()   # [{"id","label","polygon":[[x,y],...]}]
        self.parking_status: dict = {}   # slot_id -> {"occupied":bool,"confirm":int,"pending":bool|None,"changed_at":str|None}

        self.latest_frame = None          # numpy array, frame terbaru (sudah di-annotate jika AI on)
        # --- 🚀 CACHE JPEG (kunci utama stream lebih mulus) ---
        # Sebelumnya tiap request MJPEG poll (video_feed) memanggil cv2.imencode() sendiri²,
        # artinya frame yang SAMA di-encode ulang berkali-kali (poll 20x/detik vs frame baru
        # cuma ~15x/detik) — buang CPU dan malah bikin capture thread lebih lambat (makin
        # lambat capture = makin nge-jeda tampilannya). Sekarang JPEG di-encode SEKALI saja
        # tiap kali ada frame baru (di akhir loop _run), lalu di-cache di sini.
        # frame_version dipakai consumer (video_feed) buat tahu kapan ada frame BARU tanpa
        # perlu bandingkan isi gambar — cukup bandingkan angka int, murah & event-driven.
        self.latest_jpeg: Optional[bytes] = None
        self.frame_version = 0
        self.latest_alerts: list = []
        self.status = {"open": False, "error": None}
        self.frame_counter = 0
        self.lock = threading.Lock()
        self.cooldown: dict = {}
        self._last_emit = 0.0             # dipakai buat mematok TARGET_STREAM_FPS

        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.pending_capture = {"requested": False, "result": None}
        self.capture_cache: dict = {}

        # --- 📊 Metrik performa per-kamera (dipakai /api/stats/system) ---
        self.perf = {
            "fps": 0.0,
            "yolo_ms_avg": 0.0,
            "last_frame_at": 0.0,
        }
        self._fps_tick_times: collections.deque = collections.deque(maxlen=30)
        self._yolo_ms_samples: collections.deque = collections.deque(maxlen=20)

        # --- 🟩 Deteksi objek "live" (dipakai overlay kotak 3D di dashboard) ---
        # Berbeda dari latest_alerts (yang kena cooldown), ini selalu berisi
        # posisi box hasil deteksi TERBARU saja, ditimpa tiap siklus frame,
        # supaya overlay di dashboard terasa real-time tanpa numpuk histori.
        self.latest_detections: list = []

    # --- counting: load/save ke tabel camera_counts ---
    def _load_counts(self) -> dict:
        conn = get_conn()
        row = conn.execute("SELECT counts_json FROM camera_counts WHERE camera_id=?", (self.id,)).fetchone()
        conn.close()
        if row and row["counts_json"]:
            try:
                return json.loads(row["counts_json"])
            except Exception:
                pass
        return {}

    def _save_counts(self):
        conn = get_conn()
        conn.execute(
            "INSERT INTO camera_counts (camera_id, counts_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(camera_id) DO UPDATE SET counts_json=excluded.counts_json, updated_at=excluded.updated_at",
            (self.id, json.dumps(self.counts), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def reset_counts(self):
        with self.lock:
            self.counts = {}
        self._save_counts()

    def set_count_line(self, x1: float, y1: float, x2: float, y2: float):
        with self.lock:
            self.count_line = (x1, y1, x2, y2)
            # reset "side" tiap track supaya tidak salah hitung akibat garis pindah mendadak
            for tr in self.tracker.tracks.values():
                tr["side"] = None
                tr["inside"] = None

    def set_count_polygon(self, points: list):
        with self.lock:
            self.count_polygon = [[float(p[0]), float(p[1])] for p in points]
            for tr in self.tracker.tracks.values():
                tr["side"] = None
                tr["inside"] = None

    def set_count_direction(self, direction: str):
        if direction not in ("both", "in", "out"):
            direction = "both"
        with self.lock:
            self.count_direction = direction

    def set_count_shape(self, shape: str):
        if shape not in ("line", "polygon"):
            shape = "line"
        with self.lock:
            self.count_shape = shape
            for tr in self.tracker.tracks.values():
                tr["side"] = None
                tr["inside"] = None

    def set_count_line_style(self, style: Optional[str] = None, thickness: Optional[float] = None):
        with self.lock:
            if style is not None and style in ("rope", "solid", "dashed"):
                self.count_line_style = style
            if thickness is not None:
                self.count_line_thickness = max(1.0, min(6.0, float(thickness)))

    def get_counting_config(self) -> dict:
        with self.lock:
            x1, y1, x2, y2 = self.count_line
            return {
                "shape": self.count_shape,
                "direction": self.count_direction,
                "line": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "polygon": list(self.count_polygon),
                "line_style": self.count_line_style,
                "line_thickness": self.count_line_thickness,
            }

    # --- parking: load slots dari DB + status helpers ---
    def _load_parking_slots(self) -> list:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM parking_slots WHERE camera_id=? ORDER BY id ASC", (self.id,)
        ).fetchall()
        conn.close()
        slots = []
        for r in rows:
            try:
                poly = json.loads(r["polygon"])
            except Exception:
                poly = []
            slots.append({"id": r["id"], "label": r["label"] or f"Slot {r['id']}", "polygon": poly})
        return slots

    def reload_parking_slots(self):
        """Dipanggil setelah slot ditambah/diedit/dihapus lewat endpoint, supaya
        worker yang sedang jalan langsung pakai daftar slot terbaru tanpa restart."""
        fresh = self._load_parking_slots()
        with self.lock:
            self.parking_slots = fresh
            valid_ids = {s["id"] for s in fresh}
            self.parking_status = {sid: st for sid, st in self.parking_status.items() if sid in valid_ids}

    def get_parking_status(self) -> list:
        with self.lock:
            slots = list(self.parking_slots)
            status = dict(self.parking_status)
        return [
            {
                "id": s["id"], "label": s["label"], "polygon": s["polygon"],
                "occupied": status.get(s["id"], {}).get("occupied", False),
                "changed_at": status.get(s["id"], {}).get("changed_at"),
            }
            for s in slots
        ]

    def _update_parking_status(self, vehicle_centers: list, frame_w: int, frame_h: int, frame_for_alert):
        """
        Tentukan status terisi/kosong tiap slot parkir custom: SLOT dianggap terisi
        kalau ada titik tengah kendaraan (hasil YOLO siklus ini) yang jatuh di dalam
        poligon slot tsb. Pakai confirm-streak (mirip fire/smoking) supaya perubahan
        status tidak kedip-kedip akibat noise 1 siklus (mis. kendaraan sedang lewat
        di depan slot, bukan benar-benar parkir).
        """
        with self.lock:
            slots = list(self.parking_slots)

        for slot in slots:
            poly = slot.get("polygon") or []
            if len(poly) < 3:
                continue
            poly_px = [(px * frame_w, py * frame_h) for px, py in poly]
            raw_occupied = any(self._point_in_polygon(cx, cy, poly_px) for (cx, cy) in vehicle_centers)

            st = self.parking_status.setdefault(
                slot["id"], {"occupied": False, "confirm": 0, "pending": None, "changed_at": None}
            )
            if raw_occupied == st["pending"]:
                st["confirm"] += 1
            else:
                st["pending"] = raw_occupied
                st["confirm"] = 1

            if st["confirm"] >= PARKING_CONFIRM_FRAMES and st["occupied"] != raw_occupied:
                st["occupied"] = raw_occupied
                st["changed_at"] = datetime.now().isoformat()
                label = slot["label"]
                xs = [p[0] for p in poly_px]
                ys = [p[1] for p in poly_px]
                bx1, by1, bx2, by2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                if raw_occupied:
                    if self._check_cooldown(f"parking_{slot['id']}_in", sec=5):
                        lapor_deteksi_nx(
                            f"🅿️ Slot Parkir Terisi: {label} — {self.name}",
                            "Kendaraan terdeteksi menempati slot parkir.", self.nx_camera_id,
                            tags=["parking", "AI_Alert"],
                        )
                        self.latest_alerts = (
                            [self._make_alert(frame_for_alert, bx1, by1, bx2, by2, "parking", f"{label} terisi")]
                            + self.latest_alerts
                        )[:50]
                else:
                    if self._check_cooldown(f"parking_{slot['id']}_out", sec=5):
                        lapor_deteksi_nx(
                            f"🅿️ Slot Parkir Kosong: {label} — {self.name}",
                            "Kendaraan meninggalkan slot parkir.", self.nx_camera_id,
                            tags=["parking", "AI_Alert"],
                        )
                        self.latest_alerts = (
                            [self._make_alert(frame_for_alert, bx1, by1, bx2, by2, "parking", f"{label} kosong")]
                            + self.latest_alerts
                        )[:50]

    def _draw_parking_panel(self, frame):
        """Panel ringkasan kecil (pojok kanan-atas) 'PARKIR x/y terisi', diperbarui
        tiap frame dari self.parking_status — independen dari siklus YOLO."""
        with self.lock:
            slots = list(self.parking_slots)
            status = dict(self.parking_status)
        if not slots:
            return
        total = len(slots)
        occ = sum(1 for s in slots if status.get(s["id"], {}).get("occupied"))
        free = total - occ
        text = f"PARKIR {occ}/{total} terisi ({free} kosong)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        fw = frame.shape[1]
        x2 = fw - 6
        x1 = max(0, x2 - tw - 14)
        y1, y2 = 6, 6 + th + 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        color = (100, 220, 255) if occ < total else (100, 120, 255)
        cv2.putText(frame, text, (x1 + 7, y2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _line_side(self, px: float, py: float) -> float:
        """Hasil cross-product tanda (+/-) menandakan px,py ada di sisi mana dari
        garis self.count_line. Dipakai buat deteksi kapan sebuah track lintas garis."""
        x1, y1, x2, y2 = self.count_line
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    @staticmethod
    def _point_in_polygon(px: float, py: float, polygon_px: list) -> bool:
        """Ray-casting point-in-polygon test. polygon_px: list of (x,y) piksel."""
        n = len(polygon_px)
        if n < 3:
            return False
        inside = False
        x1, y1 = polygon_px[0]
        for i in range(1, n + 1):
            x2, y2 = polygon_px[i % n]
            if py > min(y1, y2) and py <= max(y1, y2) and px <= max(x1, x2) and y1 != y2:
                xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                if x1 == x2 or px <= xinters:
                    inside = not inside
            x1, y1 = x2, y2
        return inside

    def _record_crossing(self, tid: str, cls_name: str, direction: str, box: tuple, frame_for_alert):
        """Catat 1 lintasan (masuk/keluar) untuk sebuah track, dengan filter arah
        (self.count_direction) supaya kalau admin hanya mau hitung 'masuk' saja
        (atau 'keluar' saja), sisi yang tidak dipilih tidak menambah angka."""
        if self.count_direction == "in" and direction != "masuk":
            return
        if self.count_direction == "out" and direction != "keluar":
            return

        bucket = self.counts.setdefault(cls_name, {"in": 0, "out": 0})
        if direction == "masuk":
            bucket["in"] += 1
        else:
            bucket["out"] += 1
        self._save_counts()

        label_id = COUNTING_LABELS_ID.get(cls_name, cls_name)
        bx1, by1, bx2, by2 = box
        self.latest_alerts = ([self._make_alert(frame_for_alert, bx1, by1, bx2, by2, "counting", f"{label_id} {direction}")] + self.latest_alerts)[:50]
        if self._check_cooldown(f"count_{cls_name}_{direction}", sec=2):
            lapor_deteksi_nx(
                f"📊 {label_id} {direction} — {self.name}",
                f"Melintasi area/garis virtual counting ({direction}).", self.nx_camera_id,
                tags=["counting", "AI_Alert"],
            )
            kirim_metadata_analytics_nx(f"ai.counting.{cls_name}", f"{label_id} {direction}",
                                         (int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)), self.nx_camera_id)

    def _process_line_crossings(self, tracked_objects: list, frame_w: int, frame_h: int, frame_for_alert):
        """
        Dipanggil tiap siklus counting jalan. tracked_objects: list (track_id, cx, cy,
        cls_name, box) dalam koordinat piksel frame_resized.

        Mode "line": kalau sebuah track pindah sisi garis (dan sebelumnya sudah punya
        sisi tercatat), dihitung sebagai 1 lintasan — arah "masuk" kalau dari sisi
        negatif ke positif, "keluar" sebaliknya.

        Mode "polygon": kalau sebuah track berpindah status di-dalam/di-luar area,
        dihitung sebagai 1 lintasan — "masuk" kalau baru masuk ke area, "keluar"
        kalau baru keluar dari area.

        Debounce 3 detik per track supaya jitter di dekat garis/tepi area tidak
        dihitung berkali-kali. Arah yang dihitung bisa difilter lewat count_direction.
        """
        now = time.time()

        if self.count_shape == "polygon" and len(self.count_polygon) >= 3:
            polygon_px = [(px * frame_w, py * frame_h) for px, py in self.count_polygon]
            for tid, cx, cy, cls_name, box in tracked_objects:
                tr = self.tracker.tracks.get(tid)
                if tr is None:
                    continue
                is_inside = self._point_in_polygon(cx, cy, polygon_px)
                prev_inside = tr.get("inside")
                if prev_inside is not None and is_inside != prev_inside and (now - tr.get("last_counted_at", 0)) > 3.0:
                    direction = "masuk" if is_inside else "keluar"
                    tr["last_counted_at"] = now
                    self._record_crossing(tid, cls_name, direction, box, frame_for_alert)
                tr["inside"] = is_inside
            return

        # --- Mode garis (line) ---
        x1n, y1n, x2n, y2n = self.count_line
        # count_line disimpan 0-1 (relatif terhadap frame), ai settings tracker jalan
        # di resolusi frame_resized -> konversi ke piksel supaya konsisten dgn centroid.
        self.count_line_px = (x1n * frame_w, y1n * frame_h, x2n * frame_w, y2n * frame_h)
        x1, y1, x2, y2 = self.count_line_px

        for tid, cx, cy, cls_name, box in tracked_objects:
            side_val = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
            side = 1 if side_val > 0 else (-1 if side_val < 0 else 0)
            tr = self.tracker.tracks.get(tid)
            if tr is None:
                continue

            prev_side = tr.get("side")
            if prev_side is not None and side != 0 and prev_side != side and (now - tr.get("last_counted_at", 0)) > 3.0:
                direction = "masuk" if side > 0 else "keluar"
                tr["last_counted_at"] = now
                self._record_crossing(tid, cls_name, direction, box, frame_for_alert)

            if side != 0:
                tr["side"] = side

    # --- lifecycle ---
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        # Tutup paket bukti merokok yang masih menunggu post-roll. Tanpa ini,
        # alert yang menyala persis sebelum kamera dihentikan akan meninggalkan
        # folder tanpa bukti.json sama sekali.
        self.close_smoking()

    def update_ai_settings(self, **kwargs):
        with self.lock:
            self.ai_settings.update({k: v for k, v in kwargs.items() if k in self.ai_settings})

    def get_jpeg(self) -> Optional[bytes]:
        """Ambil JPEG terbaru dari cache (sudah di-encode sekali di loop _run,
        TIDAK di-encode ulang di sini) — cepat, murah, aman dipanggil sesering apapun."""
        with self.lock:
            return self.latest_jpeg

    def get_jpeg_with_version(self) -> tuple[Optional[bytes], int]:
        """Sama seperti get_jpeg(), tapi sekalian kembalikan frame_version — dipakai
        video_feed generator supaya tahu 'ini frame baru atau masih yang lama' tanpa
        perlu bandingkan isi gambar (bandingkan int jauh lebih murah)."""
        with self.lock:
            return self.latest_jpeg, self.frame_version

    def request_capture(self):
        self.pending_capture["result"] = None
        self.pending_capture["requested"] = True

    # --- AI helper logic (sama seperti versi single-camera, namun per-kamera) ---
    def _check_cooldown(self, key: str, sec: int = 8) -> bool:
        now = time.time()
        if key in self.cooldown and (now - self.cooldown[key]) < sec:
            return False
        self.cooldown[key] = now
        return True

    def _open_capture(self) -> cv2.VideoCapture:
        """Buka RTSP stream dengan timeout eksplisit (default cv2 bisa hang lama)."""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Timeout koneksi & baca, dalam ms — tanpa ini cv2 bisa menggantung lama
        # di RTSP yang unreachable (firewall block, IP salah, dll).
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Coba aktifkan hardware-accelerated decode (mis. VideoToolbox di macOS,
        # atau VAAPI/DXVA di platform lain) supaya decode H264 tidak sepenuhnya
        # membebani CPU — penting di mesin dual-core tanpa GPU dedicated seperti
        # MacBook Air 2017. Aman di-skip kalau build OpenCV tidak mendukungnya.
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except Exception:
            pass
        return cap

    def _run(self):
        cap = self._open_capture()

        # cv2.isOpened() bisa True meski RTSP sebenarnya tidak reachable
        # (FFMPEG backend lazy-connect) — makanya kita wajib test-read dulu.
        opened = cap.isOpened()
        test_ok = False
        if opened:
            test_ok, _ = cap.read()

        if not opened or not test_ok:
            self.status["open"] = False
            self.status["error"] = (
                "RTSP URL tidak bisa dibuka (cek IP/port/firewall)." if not opened
                else "RTSP terbuka tapi tidak ada frame masuk (cek username/password/path channel)."
            )
            cap.release()
            print(f"❌ [{self.name}] Gagal konek RTSP: {self.status['error']} → {self.rtsp_url}")
            return

        self.status["open"] = True
        self.status["error"] = None
        print(f"✅ [{self.name}] RTSP terhubung: {self.rtsp_url}")
        consecutive_failures = 0

        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print(f"⚠️ [{self.name}] Stream terputus. Reconnecting…")
                    self.status["open"] = False
                    self.status["error"] = "Stream terputus, mencoba reconnect…"
                    cap.release()
                    time.sleep(3)
                    cap = self._open_capture()
                    if cap.isOpened():
                        test_ok, _ = cap.read()
                        if test_ok:
                            self.status["open"] = True
                            self.status["error"] = None
                            print(f"✅ [{self.name}] Reconnect berhasil.")
                        else:
                            self.status["error"] = "Reconnect gagal: stream terbuka tapi tidak ada frame."
                    else:
                        self.status["error"] = "Reconnect gagal: RTSP tidak bisa dibuka."
                    consecutive_failures = 0
                else:
                    time.sleep(0.1)
                continue

            consecutive_failures = 0

            # --- ⏱️ Patok FPS stream (mis. 15fps) ---
            # Kamera IP biasanya kirim 20-30fps mentah. Kalau semua frame diproses
            # (resize + gambar overlay + encode JPEG), CPU dual-core cepat penuh.
            # Frame yang datang lebih cepat dari target cukup "dibuang" di sini —
            # decode tetap terjadi (tidak bisa dihindari di OpenCV/FFMPEG), tapi kerja
            # berat sesudahnya (AI, overlay, encode) tidak dikerjakan untuk frame yg dibuang.
            now_pace = time.time()
            min_interval = 1.0 / TARGET_STREAM_FPS if TARGET_STREAM_FPS > 0 else 0
            if min_interval and (now_pace - self._last_emit) < min_interval:
                continue
            self._last_emit = now_pace

            ai_res = (480, 270) if LOW_POWER_MODE else (640, 360)
            frame_resized = cv2.resize(frame, ai_res)
            frame_h, frame_w = frame_resized.shape[:2]
            self.frame_counter += 1
            current_alerts: list = []
            current_detections: list = []

            # --- 📊 Update FPS berjalan (rolling window ~30 frame terakhir) ---
            now_tick = time.time()
            self._fps_tick_times.append(now_tick)
            self.perf["last_frame_at"] = now_tick
            if len(self._fps_tick_times) >= 2:
                span = self._fps_tick_times[-1] - self._fps_tick_times[0]
                if span > 0:
                    self.perf["fps"] = round((len(self._fps_tick_times) - 1) / span, 1)

            with self.lock:
                ai = dict(self.ai_settings)  # snapshot supaya tidak race saat toggle

            # --- Capture request (untuk registrasi wajah) ---
            if self.pending_capture["requested"]:
                faces_now = self._detect_faces_full(frame_resized)
                if len(faces_now) == 1:
                    d0 = faces_now[0]
                    (x, y, w, h) = d0["box"]
                    # Simpan crop yang SUDAH di-align (mata di posisi kanonik),
                    # bukan potongan kotak mentah. Kotak mentah membuat skala &
                    # kemiringan tiap pendaftaran berbeda-beda, dan itulah salah
                    # satu sebab "sudah terdaftar tapi tidak dikenali".
                    self.pending_capture["result"] = {
                        "ok": True,
                        "face_crop": face_engine.align(frame_resized, d0),
                        "frame_crop": frame_resized[max(0, y):y + h, max(0, x):x + w].copy(),
                    }
                elif len(faces_now) == 0:
                    self.pending_capture["result"] = {"ok": False, "reason": "Tidak ada wajah terdeteksi."}
                else:
                    self.pending_capture["result"] = {"ok": False, "reason": "Lebih dari satu wajah terdeteksi."}
                self.pending_capture["requested"] = False

            # --- 🧹 Bersihkan langsung track dari fitur yang SEDANG dimatikan admin,
            # supaya box lama tidak nyangkut selamanya di overlay. ---
            disabled_types = set()
            if not ai["fire_detection"]:
                disabled_types.add("fire")
            if not ai["smoking_detection"]:
                disabled_types.add("smoking")
            if not ai["behavior_detection"]:
                disabled_types.add("behavior_fall")
            if not ai["face_recognition"]:
                disabled_types.add("face")
            self.overlay_tracker.purge_types(disabled_types)

            # --- 📊 Panel angka akumulasi counting (in/out) tetap tampil di live view
            # (informasi berguna & kecil, tidak mengganggu). Garis/area virtualnya SENGAJA
            # TIDAK digambar di sini lagi — itu cuma tampil interaktif di layar Setting
            # Counting (SVG client-side), supaya live monitoring bersih tidak terganggu. ---
            run_counting = ai["people_counting"] or ai["vehicle_counting"] or ai["animal_counting"]
            if not run_counting:
                self._last_tracked_objects = []
            if run_counting:
                self._draw_count_overlay_panel(frame_resized, ai)
            if ai["parking_detection"]:
                self._draw_parking_panel(frame_resized)

            # --- 🔥 Fire/Smoke: engine warna (HSV), independen dari siklus YOLO, cadence sendiri ---
            if ai["fire_detection"] and self.frame_counter % FIRE_EVERY_N_FRAMES == 0:
                try:
                    fire_regions = self._detect_fire_smoke_regions(frame_resized, frame_w, frame_h)
                    fire_dets = [
                        {"cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2, "type": "fire", "box": (x1, y1, x2, y2),
                         "label": "Api" if label == "Indikasi Api" else "Asap"}
                        for (x1, y1, x2, y2, label, _area) in fire_regions
                    ]
                    fire_tracked = self.overlay_tracker.update(fire_dets, families={"fire"})
                    for t in fire_tracked:
                        if t["confirm"] == FIRE_CONFIRM_FRAMES and self._check_cooldown("fire"):
                            x1, y1, x2, y2 = t["box"]
                            lapor_deteksi_nx(
                                f"🔥 Bahaya Kebakaran — {self.name}",
                                f"Indikasi: {t['label']}", self.nx_camera_id,
                                tags=["fire", "critical", "AI_Alert"],
                            )
                            kirim_metadata_analytics_nx("ai.fire.object", t["label"],
                                                         (x1, y1, x2 - x1, y2 - y1), self.nx_camera_id)
                            current_alerts.append(self._make_alert(frame_resized, x1, y1, x2, y2, "fire", f"Indikasi {t['label']}"))
                except Exception as e:
                    self.status["error"] = f"Fire detection error: {e}"

            # --- YOLO: smoking (heuristik spasial) / behavior / people & vehicle counting ---
            # Semua fitur ini pakai 1x inferensi YOLO yang sama per siklus (hemat CPU),
            # lalu tiap box hasil deteksi diperiksa untuk kategori mana yang relevan.
            run_yolo = ai["smoking_detection"] or ai["behavior_detection"] or run_counting or ai["parking_detection"]
            if run_yolo and self.frame_counter % YOLO_EVERY_N_FRAMES == 0:
                try:
                    _yolo_t0 = time.time()
                    with yolo_lock:
                        # device=YOLO_DEVICE -> paksa GPU kalau tersedia (bukan cuma berharap
                        # ultralytics auto-detect). half=True (FP16) hanya valid di GPU, dan
                        # diabaikan otomatis oleh ultralytics kalau model sumbernya sudah .engine
                        # (TensorRT engine presisinya sudah tetap dari saat export).
                        results = yolo_model.predict(
                            frame_resized, conf=0.35, imgsz=YOLO_IMG_SIZE,
                            device=YOLO_DEVICE, half=CUDA_AVAILABLE, verbose=False,
                        )
                    _yolo_ms = (time.time() - _yolo_t0) * 1000
                    self._yolo_ms_samples.append(_yolo_ms)
                    self.perf["yolo_ms_avg"] = round(sum(self._yolo_ms_samples) / len(self._yolo_ms_samples), 1)
                    counting_dets: list = []       # (cx, cy, category, box) buat SimpleTracker (counting)
                    person_boxes: list = []        # semua box "person" siklus ini (buat heuristik merokok)
                    smoking_obj_boxes: list = []   # semua box objek kandidat merokok siklus ini
                    behavior_dets: list = []       # buat OverlayTracker family "behavior_fall"
                    parking_vehicle_centers: list = []  # (cx, cy) semua kendaraan siklus ini, buat cek slot parkir

                    for r in results:
                        for box in r.boxes:
                            cls_name = yolo_model.names[int(box.cls[0])]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w_box, h_box = x2 - x1, y2 - y1

                            if cls_name == "person":
                                person_boxes.append((x1, y1, x2, y2))
                            if ai["smoking_detection"] and cls_name in SMOKING_CONFUSER_CLASSES:
                                # Dikumpulkan sebagai PENYANGKAL — kebalikan dari
                                # versi lama yang memakainya sebagai bukti merokok.
                                smoking_obj_boxes.append((x1, y1, x2, y2, cls_name))
                            if ai["parking_detection"] and cls_name in VEHICLE_CLASSES:
                                parking_vehicle_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

                            if ai["behavior_detection"] and cls_name == "person" and w_box > (h_box * 1.2):
                                behavior_dets.append({
                                    "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2, "type": "behavior_fall",
                                    "box": (x1, y1, x2, y2), "label": "",
                                })

                            # --- 🧍🚗🐾 Klasifikasi People / Vehicle(mobil,motor,bus,truk) / Hewan ---
                            category = None
                            if cls_name == "person" and ai["people_counting"]:
                                category = "person"
                            elif cls_name in VEHICLE_CLASSES and ai["vehicle_counting"]:
                                category = cls_name
                            elif cls_name in ANIMAL_CLASSES and ai["animal_counting"]:
                                category = "animal"
                            if category:
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                counting_dets.append((cx, cy, category, (x1, y1, x2, y2)))

                    # --- 🚬 Deteksi merokok multi-bukti ---
                    # Engine menilai TIAP ORANG secara terpisah dan lintas waktu:
                    #   B1 pola siklus isapan (naik-tahan 1-4 dtk-turun, berulang)
                    #   B2 kepulan asap yang muncul-hilang di sekitar kepala
                    #   B3 bara kecil terang di zona mulut (hanya saat redup)
                    #   B4 HP/gelas/botol terlihat -> MENGURANGI skor
                    # Alert hanya menyala kalau skor gabungan lewat ambang DAN ada
                    # minimal satu bukti yang kuat sendirian.
                    if ai["smoking_detection"]:
                        try:
                            smoking_events = self.smoking_engine.update(
                                frame_resized, person_boxes, smoking_obj_boxes)

                            # Overlay diambil dari snapshot engine supaya kotak tetap
                            # menempel pada orangnya walau siklus ini belum ada alert.
                            self.overlay_tracker.update([
                                {"cx": (b["box"][0] + b["box"][2]) // 2,
                                 "cy": (b["box"][1] + b["box"][3]) // 2,
                                 "type": "smoking", "box": b["box"],
                                 "label": f"Merokok {b['score']:.0%}" if b["score"] >= smoking_cfg.SCORE_THRESHOLD
                                          else f"Memantau {b['isapan']} isapan"}
                                for b in self.smoking_engine.snapshot()
                            ], families={"smoking"})

                            for ev in smoking_events:
                                x1, y1, x2, y2 = ev["box"]
                                c = ev["cues"]
                                rincian = (f"gestur {c['gestur_siklik']:.2f} "
                                           f"({c['isapan']} isapan), asap {c['kepulan_asap']:.2f}, "
                                           f"bara {c['bara']:.2f}, penyangkal {c['penyangkal_terlihat']:.2f}")
                                lapor_deteksi_nx(
                                    f"🚬 Terdeteksi Merokok ({ev['score']:.0%}) — {self.name}",
                                    rincian, self.nx_camera_id,
                                    tags=["smoking", "AI_Alert"],
                                )
                                kirim_metadata_analytics_nx("ai.smoking.object", ev["label"],
                                                             (x1, y1, x2 - x1, y2 - y1), self.nx_camera_id)
                                alert = self._make_alert(frame_resized, x1, y1, x2, y2,
                                                         "smoking", ev["label"])
                                # Alert dibawa bersama rincian buktinya, supaya operator
                                # bisa menilai sendiri alih-alih menerima kata "indikasi"
                                # tanpa dasar seperti versi lama.
                                alert["cues"] = c
                                alert["evidence_id"] = ev["evidence_id"]
                                current_alerts.append(alert)
                        except Exception as e:
                            self.status["error"] = f"Smoking detection error: {e}"

                    if ai["behavior_detection"]:
                        behavior_tracked = self.overlay_tracker.update(behavior_dets, families={"behavior_fall"})
                        for t in behavior_tracked:
                            if t["confirm"] == BEHAVIOR_CONFIRM_FRAMES and self._check_cooldown("fall"):
                                x1, y1, x2, y2 = t["box"]
                                lapor_deteksi_nx(
                                    f"🚨 Anomali: Orang Terjatuh — {self.name}",
                                    "Deteksi posisi horizontal", self.nx_camera_id,
                                    tags=["fall", "critical", "AI_Alert"],
                                )
                                kirim_metadata_analytics_nx("ai.fall.object", "Orang Terjatuh",
                                                             (x1, y1, x2 - x1, y2 - y1), self.nx_camera_id)
                                current_alerts.append(self._make_alert(frame_resized, x1, y1, x2, y2, "behavior_fall", "Orang Terjatuh / Terkapar"))

                    if run_counting:
                        tracked_objects = self.tracker.update(counting_dets)
                        self._process_line_crossings(tracked_objects, frame_w, frame_h, frame_resized)
                        self._last_tracked_objects = tracked_objects

                    if ai["parking_detection"]:
                        self._update_parking_status(parking_vehicle_centers, frame_w, frame_h, frame_resized)
                except Exception as e:
                    self.status["error"] = f"YOLO error: {e}"

            # --- Face recognition ---
            # Dulu jalan di SETIAP frame (mahal). Sekarang di-skip seperti YOLO supaya
            # CPU jauh lebih ringan, tetap cukup responsif untuk kebutuhan surveillance.
            if ai["face_recognition"] and self.frame_counter % FACE_EVERY_N_FRAMES == 0:
                try:
                    faces = self._detect_faces_full(frame_resized)
                    face_dets = []
                    for d in faces:
                        (x, y, w, h) = d["box"]
                        name, occ = None, "?"
                        if recognizer_ready:
                            try:
                                res = face_engine.recognize_frame(frame_resized, d)
                                name = res.get("name")
                                occ = res.get("occlusion", {}).get("state", "?")
                            except Exception:
                                pass
                        face_dets.append({
                            "cx": x + w // 2, "cy": y + h // 2, "type": "face",
                            "box": (x, y, x + w, y + h),
                            "label": name or "Tidak Dikenal",
                            "_name": name, "_occ": occ,
                        })

                    face_tracked = self.overlay_tracker.update(face_dets, families={"face"})
                    # Track yang benar-benar di-refresh siklus ini dikenali dari
                    # centroid-nya (tracker menyalin cx/cy apa adanya). Hanya track
                    # itu yang boleh menyumbang suara — kalau tidak, track yang
                    # sedang "menempel" tanpa deteksi baru akan mengulang suara lama
                    # terus-menerus dan voting jadi tidak ada artinya.
                    fresh = {(d["cx"], d["cy"]): d for d in face_dets}
                    for t in face_tracked:
                        key = (t["cx"], t["cy"])
                        if key in fresh:
                            voted = self.face_voter.push(t["track_id"], fresh[key]["_name"])
                        else:
                            voted = self.face_voter.get(t["track_id"])

                        tr = self.overlay_tracker.tracks.get(t["track_id"])
                        if tr is not None:
                            tr["label"] = voted or "Tidak Dikenal"
                        t["label"] = voted or "Tidak Dikenal"

                        if voted and self._check_cooldown(f"face_{voted}"):
                            x1, y1, x2, y2 = t["box"]
                            occ_txt = {"mask": " (bermasker)",
                                       "headwear": " (bertopi/berkerudung)",
                                       "mask_headwear": " (bermasker + bertopi/berkerudung)"}.get(
                                           fresh.get(key, {}).get("_occ"), "")
                            lapor_deteksi_nx(
                                f"👤 Wajah Dikenali: {voted}{occ_txt} — {self.name}",
                                "Wajah terdaftar terdeteksi", self.nx_camera_id,
                                tags=["face_recognition", "AI_Alert"],
                            )
                            kirim_metadata_analytics_nx("ai.face.object", voted, (x1, y1, x2 - x1, y2 - y1), self.nx_camera_id)
                            current_alerts.append(self._make_alert(frame_resized, x1, y1, x2, y2, "face", voted))

                    self.face_voter.purge({t["track_id"] for t in face_tracked})
                except Exception as e:
                    self.status["error"] = f"Face recognition error: {e}"

            # --- 🧊 Rebuild current_detections TIAP FRAME dari snapshot tracker (bukan cuma
            # pas siklus AI jalan) — inilah kunci box "mengikuti motion" mulus alih-alih
            # kedip muncul/hilang tiap kali YOLO/cascade sempat jalan. OverlayTracker &
            # counting tracker sama-sama menyimpan posisi terakhir sampai beberapa siklus
            # missed berturut, jadi box tetap ada & mulus dianimasikan client-side. ---
            for t in self.overlay_tracker.snapshot():
                x1, y1, x2, y2 = t["box"]
                threshold = {
                    # Merokok tidak lagi memakai confirm-streak: keyakinannya sudah
                    # dinilai engine lewat skor multi-bukti, jadi kotaknya boleh langsung
                    # tampil (label sudah membedakan "Memantau" vs "Merokok").
                    "fire": FIRE_CONFIRM_FRAMES, "smoking": 1,
                    "behavior_fall": BEHAVIOR_CONFIRM_FRAMES, "face": 1,
                }.get(t["type"], 1)
                display_label = t["label"] if t["confirm"] >= threshold else "Memeriksa…"
                current_detections.append({
                    **self._norm_box(x1, y1, x2, y2, frame_w, frame_h, t["type"], display_label),
                    "track_id": f"{t['type']}_{t['track_id']}",
                    "confirmed": t["confirm"] >= threshold,
                })
            for (tid, cx, cy, cls_name, box) in getattr(self, "_last_tracked_objects", []):
                x1, y1, x2, y2 = box
                label_id = COUNTING_LABELS_ID.get(cls_name, cls_name)
                current_detections.append({
                    **self._norm_box(x1, y1, x2, y2, frame_w, frame_h, f"count_{cls_name}", label_id),
                    "track_id": f"count_{cls_name}_{tid}",
                    "confirmed": True,
                })

            # --- 📦 Encode JPEG SEKALI per frame baru (di luar lock supaya lock dipegang
            # sesingkat mungkin). Kualitas 80 cukup tajam untuk live view tapi jauh lebih
            # kecil/cepat di-encode & dikirim dibanding kualitas default (~95), yang mana
            # ini juga membantu stream lebih mulus di jaringan/CPU terbatas. ---
            ok_jpg, jpg_buf = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpg_buf.tobytes() if ok_jpg else None

            with self.lock:
                self.latest_frame = frame_resized.copy()
                self.latest_detections = current_detections
                if current_alerts:
                    self.latest_alerts = (current_alerts + self.latest_alerts)[:50]
                if jpeg_bytes is not None:
                    self.latest_jpeg = jpeg_bytes
                    self.frame_version += 1

        cap.release()
        self.status["open"] = False

    @staticmethod
    def _detect_fire_smoke_regions(frame_bgr, frame_w, frame_h):
        """
        Deteksi indikasi api & asap berbasis analisis warna (HSV) + luas area minimum.
        Tanpa perlu model AI tambahan (ringan di CPU manapun) — akurasi ditingkatkan
        lewat FIRE_SENSITIVITY (rentang warna) + FIRE_MIN_AREA_PCT (buang noise kecil)
        + confirm-streak multi-siklus di OverlayTracker (buang false-positive sekali kedip,
        mis. pantulan lampu, baju oranye, dsb yang biasanya tidak konsisten antar siklus).

        Return: list of (x1, y1, x2, y2, label, area_px)
        """
        s = max(1, min(10, FIRE_SENSITIVITY))
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # --- Warna api: hue rendah (merah-oranye-kuning), saturasi & value tinggi/menyala ---
        sat_min = max(60, 150 - s * 9)
        val_min = max(130, 215 - s * 8)
        mask_fire = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([30, 255, 255]))
        # sebagian api condong ke arah hue tinggi mendekati merah murni (wrap-around hue 170-180)
        mask_fire2 = cv2.inRange(hsv, np.array([170, sat_min, val_min]), np.array([180, 255, 255]))
        mask_fire = cv2.bitwise_or(mask_fire, mask_fire2)

        # --- Warna asap: saturasi rendah (abu-abu/putih pudar), value menengah-tinggi ---
        sat_max_smoke = max(18, 55 - s * 3)
        mask_smoke = cv2.inRange(hsv, np.array([0, 0, 95]), np.array([180, sat_max_smoke, 230]))

        kernel = np.ones((5, 5), np.uint8)
        mask_fire = cv2.morphologyEx(mask_fire, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_fire = cv2.dilate(mask_fire, kernel, iterations=2)
        mask_smoke = cv2.morphologyEx(mask_smoke, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_smoke = cv2.dilate(mask_smoke, kernel, iterations=1)

        min_area = max(50, int(frame_w * frame_h * (FIRE_MIN_AREA_PCT / 100.0)))
        results = []
        for mask, label in ((mask_fire, "Indikasi Api"), (mask_smoke, "Indikasi Asap")):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # buang bentuk terlalu pipih (biasanya garis/pantulan cahaya, bukan kobaran/gumpalan)
                if w == 0 or h == 0 or max(w / h, h / w) > 6:
                    continue
                results.append((x, y, x + w, y + h, label, area))
        return results

    @staticmethod
    def _detect_faces_full(frame_bgr):
        """Deteksi wajah lewat engine (YuNet -> Haar frontal -> Haar profil).

        DIUBAH TOTAL dari versi Haar-saja. Haar frontal dilatih pakai wajah utuh,
        jadi masker/topi/kerudung membuatnya GAGAL MENEMUKAN WAJAH sama sekali —
        dan kalau wajah tidak terdeteksi, secanggih apapun recognizer-nya tidak
        akan pernah dipanggil. Pada uji internal, Haar gagal mendeteksi 8 dari 8
        wajah bermasker. YuNet dilatih di WIDER FACE yang penuh wajah terhalang,
        dan sebagai bonus mengembalikan titik mata yang dipakai untuk alignment.

        Return: list dict deteksi (punya 'box' dan 'landmarks').
        """
        return face_engine.detect(frame_bgr)

    @staticmethod
    def _detect_faces_fast(gray_full):
        """Kompatibilitas mundur: kembalikan hanya kotak (x, y, w, h)."""
        src = gray_full if gray_full.ndim == 3 else cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        return [d["box"] for d in face_engine.detect(src)]

    @staticmethod
    def _draw_box(frame, x1, y1, x2, y2, color, label):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    @staticmethod
    def _draw_rope_segment(frame, pt1, pt2, color, thickness: float = 2.0):
        """
        Gambar 1 ruas garis dengan gaya "tali" (rope) — dua untai tipis yang
        saling melintir (twisted strands) dibanding garis solid biasa, supaya
        bisa dibuat lebih kecil/halus (thickness diperkecil) tapi tetap jelas
        terlihat sebagai garis virtual, bukan cuma garis lurus polos.
        """
        import math
        x1, y1 = pt1
        x2, y2 = pt2
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length   # arah sepanjang garis
        nx, ny = -uy, ux                                    # arah tegak lurus (normal)
        amp = max(1.5, thickness * 1.3)                     # amplitudo "lintiran" tali
        period = max(8.0, thickness * 6.0)                  # panjang 1 puntiran
        steps = max(2, int(length // 3))
        light = tuple(min(255, int(c * 1.35) + 40) for c in color)

        prev1 = prev2 = None
        for i in range(steps + 1):
            t = i / steps
            d = t * length
            phase = (d / period) * 2 * math.pi
            off = math.sin(phase) * amp
            px = x1 + ux * d + nx * off
            py = y1 + uy * d + ny * off
            px2 = x1 + ux * d - nx * off
            py2 = y1 + uy * d - ny * off
            cur1 = (int(px), int(py))
            cur2 = (int(px2), int(py2))
            if prev1 is not None:
                cv2.line(frame, prev1, cur1, color, max(1, int(round(thickness))))
                cv2.line(frame, prev2, cur2, light, max(1, int(round(thickness * 0.7))))
            prev1, prev2 = cur1, cur2

    def _draw_counting_zone(self, frame, frame_w, frame_h):
        """Gambar garis virtual (mode 'line') atau area/poligon (mode 'polygon')
        sesuai konfigurasi kamera ini, dengan gaya sesuai count_line_style.

        CATATAN: method ini SENGAJA TIDAK dipanggil lagi dari _run() (live stream)
        supaya garis virtual tidak mengganggu tampilan pemantauan sehari-hari.
        Editor garis interaktif di dashboard (modal Counting) menggambar preview-nya
        sendiri di sisi client pakai SVG di atas <img> live feed, jadi garis HANYA
        terlihat saat admin sedang membuka layar Setting Counting. Method ini
        dibiarkan ada (tidak dihapus) untuk kemungkinan pemakaian lain di masa depan
        (mis. overlay pada rekaman/klip yang diunduh)."""
        color = (0, 255, 255)
        style = self.count_line_style
        thickness = self.count_line_thickness
        dir_label = {"both": "MASUK+KELUAR", "in": "MASUK SAJA", "out": "KELUAR SAJA"}.get(self.count_direction, "")

        if self.count_shape == "polygon" and len(self.count_polygon) >= 3:
            pts_px = [(int(px * frame_w), int(py * frame_h)) for px, py in self.count_polygon]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(pts_px, dtype=np.int32)], (0, 200, 255))
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, dst=frame)
            n = len(pts_px)
            for i in range(n):
                p1, p2 = pts_px[i], pts_px[(i + 1) % n]
                if style == "rope":
                    self._draw_rope_segment(frame, p1, p2, color, thickness)
                else:
                    cv2.line(frame, p1, p2, color, max(1, int(round(thickness))),
                             lineType=cv2.LINE_AA if style == "solid" else cv2.LINE_4)
            for p in pts_px:
                cv2.circle(frame, p, max(3, int(thickness * 2)), color, -1)
            label_pt = pts_px[0]
            cv2.putText(frame, f"AREA COUNTING ({dir_label})", (label_pt[0], max(12, label_pt[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        else:
            lx1, ly1, lx2, ly2 = self.count_line
            pt1 = (int(lx1 * frame_w), int(ly1 * frame_h))
            pt2 = (int(lx2 * frame_w), int(ly2 * frame_h))
            if style == "rope":
                self._draw_rope_segment(frame, pt1, pt2, color, thickness)
            else:
                cv2.line(frame, pt1, pt2, color, max(1, int(round(thickness))),
                         lineType=cv2.LINE_AA if style == "solid" else cv2.LINE_4)
            cv2.circle(frame, pt1, max(3, int(thickness * 2)), color, -1)
            cv2.circle(frame, pt2, max(3, int(thickness * 2)), color, -1)
            cv2.putText(frame, f"COUNT LINE ({dir_label})", (pt1[0], max(12, pt1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def _draw_count_overlay_panel(self, frame, ai: dict):
        """
        Panel angka akumulasi real-time (masuk/keluar) di pojok kiri-atas frame,
        supaya jumlah orang/kendaraan/hewan yang sudah tercatat langsung terlihat
        di layar CCTV tanpa perlu buka dashboard. Diperbarui tiap frame dari
        self.counts (state yang di-update oleh _record_crossing), independen dari
        siklus YOLO, jadi selalu menampilkan angka terbaru.
        """
        with self.lock:
            counts_snapshot = {k: dict(v) for k, v in self.counts.items()}

        rows = []
        if ai.get("people_counting"):
            c = counts_snapshot.get("person", {"in": 0, "out": 0})
            rows.append(("Orang", c.get("in", 0), c.get("out", 0), (46, 204, 113)))
        if ai.get("vehicle_counting"):
            v_in = sum(counts_snapshot.get(k, {}).get("in", 0) for k in VEHICLE_CLASSES)
            v_out = sum(counts_snapshot.get(k, {}).get("out", 0) for k in VEHICLE_CLASSES)
            rows.append(("Kendaraan", v_in, v_out, (255, 140, 0)))
        if ai.get("animal_counting"):
            c = counts_snapshot.get("animal", {"in": 0, "out": 0})
            rows.append(("Hewan", c.get("in", 0), c.get("out", 0), (255, 193, 7)))

        if not rows:
            return

        pad = 8
        row_h = 18
        panel_w = 168
        panel_h = pad * 2 + row_h * len(rows)
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        cv2.rectangle(frame, (6, 6), (6 + panel_w, 6 + panel_h), (0, 255, 255), 1)

        for i, (label, n_in, n_out, color) in enumerate(rows):
            y = 6 + pad + row_h * i + 13
            cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"IN {n_in}", (92, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(frame, f"OUT {n_out}", (135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 150, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _norm_box(x1, y1, x2, y2, frame_w, frame_h, box_type, label):
        """Normalisasi koordinat box ke rentang 0-1 supaya dashboard bisa menggambar
        overlay 3D di atas <img> stream berapa pun ukuran tampil/aslinya."""
        return {
            "type": box_type,
            "label": label,
            "x1": round(max(0, x1) / frame_w, 4),
            "y1": round(max(0, y1) / frame_h, 4),
            "x2": round(min(frame_w, x2) / frame_w, 4),
            "y2": round(min(frame_h, y2) / frame_h, 4),
        }

    def close_smoking(self):
        """Tutup paket bukti yang masih menunggu post-roll saat kamera berhenti."""
        try:
            self.smoking_engine.close()
        except Exception:
            pass

    def _make_alert(self, frame, x1, y1, x2, y2, alert_type, name):
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            crop = frame
        _, buf = cv2.imencode(".jpg", crop)
        b64 = base64.b64encode(buf).decode()

        # --- simpan ke disk supaya bisa di-browse/"diputar ulang" lewat galeri capture ---
        # (di luar cooldown/memori latest_alerts yang cuma nyimpan 50 terakhir & hilang
        # saat restart). Nama file diberi suffix uuid pendek supaya tidak bentrok kalau
        # ada >1 alert di detik yang sama.
        capture_filename = None
        try:
            now = datetime.now()
            ts = now.strftime("%Y%m%d_%H%M%S")
            safe_type = re.sub(r"[^a-z_]", "", alert_type.lower()) or "event"
            fname = f"{ts}_{safe_type}_{uuid.uuid4().hex[:6]}.jpg"
            cam_dir = os.path.join(CAPTURES_DIR_DEFAULT, str(self.id))
            os.makedirs(cam_dir, exist_ok=True)
            fpath = os.path.join(cam_dir, fname)
            with open(fpath, "wb") as f:
                f.write(buf.tobytes())
            with open(fpath[:-4] + ".json", "w") as f:
                json.dump({
                    "type": alert_type, "name": name, "camera_id": self.id,
                    "camera_name": self.name, "created_at": now.isoformat(),
                }, f)
            capture_filename = fname
        except Exception as e:
            print(f"⚠️  [{self.name}] Gagal menyimpan capture ke disk: {e}")

        return {
            "type": alert_type, "name": name, "time": time.strftime("%H:%M:%S"),
            "image": "data:image/jpeg;base64," + b64,
            "capture_file": capture_filename,
        }


class StorageManager:
    """
    Kelola lokasi penyimpanan rekaman: admin bisa atur beberapa lokasi (disk/folder
    berbeda) dengan urutan prioritas (primary, cadangan 1, cadangan 2, dst).

    Background thread (_storage_monitor_loop) cek kesehatan lokasi aktif tiap
    STORAGE_HEALTH_CHECK_INTERVAL_SEC detik: apakah masih bisa ditulis, dan apakah
    sisa ruang masih di atas MIN_FREE_SPACE_GB. Begitu lokasi aktif bermasalah,
    OTOMATIS pindah (failover) ke lokasi berikutnya yang sehat sesuai prioritas —
    tanpa perlu campur tangan admin. Kalau lokasi prioritas lebih tinggi pulih lagi
    dan STABIL sehat (bukan cuma sesaat), otomatis pindah balik ke situ juga.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.paths: list = []          # urutan prioritas, hasil load dari DB
        self.active_path: Optional[str] = None
        self.active_index: int = -1
        self._healthy_streak: dict = {}
        self.event_log = collections.deque(maxlen=50)
        self._load_paths()
        self._ensure_default_path()

    def _load_paths(self):
        conn = get_conn()
        rows = conn.execute("SELECT * FROM storage_paths WHERE enabled=1 ORDER BY priority ASC").fetchall()
        conn.close()
        with self.lock:
            self.paths = [dict(r) for r in rows]

    def _ensure_default_path(self):
        """Instalasi baru / belum pernah atur lokasi sama sekali -> pakai folder
        default di sebelah server.py sbg primary, biar tetap jalan out-of-the-box."""
        if self.paths:
            return
        os.makedirs(RECORDINGS_DIR_DEFAULT, exist_ok=True)
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO storage_paths (path, priority, label) VALUES (?, 0, ?)",
                (RECORDINGS_DIR_DEFAULT, "Default (di sebelah server)"),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # sudah ada (race kondisi start bersamaan), abaikan
        conn.close()
        self._load_paths()

    @staticmethod
    def check_health(path: str) -> dict:
        """Cek 1 lokasi: folder bisa dibuat/ditulis + sisa ruang masih cukup."""
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".enpidix_write_test")
            with open(test_file, "wb") as f:
                f.write(b"ok")
            os.remove(test_file)
        except Exception as e:
            return {"healthy": False, "reason": f"Tidak bisa ditulis ({e})", "free_gb": None, "total_gb": None}

        free_gb = total_gb = None
        try:
            usage = shutil.disk_usage(path)
            free_gb = round(usage.free / (1024 ** 3), 2)
            total_gb = round(usage.total / (1024 ** 3), 2)
        except Exception:
            pass

        if free_gb is not None and free_gb < MIN_FREE_SPACE_GB:
            return {"healthy": False, "reason": f"Sisa ruang cuma {free_gb} GB (ambang minimum {MIN_FREE_SPACE_GB} GB)",
                    "free_gb": free_gb, "total_gb": total_gb}
        return {"healthy": True, "reason": None, "free_gb": free_gb, "total_gb": total_gb}

    def _switch_to(self, idx: int, p: dict, reason: str):
        changed = (self.active_path != p["path"])
        self.active_path = p["path"]
        self.active_index = idx
        if changed:
            self._log_event(f"🔀 Storage aktif pindah ke \"{p['label'] or p['path']}\" (prioritas #{idx + 1}) — {reason}")
        return changed

    def _log_event(self, msg: str):
        self.event_log.append({"ts": datetime.now().isoformat(), "message": msg})
        print(f"💾 [Storage] {msg}")

    def recheck(self) -> bool:
        """Jalankan 1 siklus health-check + failover/failback kalau perlu.
        Return True kalau lokasi aktif BERUBAH siklus ini (dipakai caller untuk
        tahu kapan perlu restart proses ffmpeg yang sedang berjalan)."""
        with self.lock:
            if not self.paths:
                self._ensure_default_path()
                if not self.paths:
                    return False

            results = []
            for idx, p in enumerate(self.paths):
                h = self.check_health(p["path"])
                results.append((idx, p, h))
                key = p["path"]
                self._healthy_streak[key] = self._healthy_streak.get(key, 0) + 1 if h["healthy"] else 0

            current_healthy = False
            if 0 <= self.active_index < len(results):
                _, p_cur, h_cur = results[self.active_index]
                current_healthy = h_cur["healthy"]

            before = self.active_path

            if current_healthy:
                # Sudah sehat di posisi sekarang — cek apakah ada prioritas LEBIH TINGGI
                # yg sudah pulih & stabil, kalau iya pindah balik (fail-back terkendali).
                for idx, p, h in results:
                    if idx >= self.active_index:
                        break
                    if h["healthy"] and self._healthy_streak[p["path"]] >= STORAGE_RECOVERY_STABLE_CHECKS:
                        self._switch_to(idx, p, "lokasi prioritas lebih tinggi sudah pulih & stabil")
                        break
            else:
                # Lokasi aktif bermasalah -> failover SEGERA ke yang sehat berikutnya,
                # tidak perlu tunggu stabil (situasi darurat, rekaman harus tetap jalan).
                switched = False
                for idx, p, h in results:
                    if h["healthy"]:
                        self._switch_to(idx, p, "lokasi sebelumnya bermasalah")
                        switched = True
                        break
                if not switched and self.paths:
                    p0 = self.paths[0]
                    if self.active_path != p0["path"]:
                        self._log_event("🚨 SEMUA lokasi penyimpanan bermasalah! Rekaman kemungkinan gagal menulis sampai salah satu disk pulih.")
                    self.active_path, self.active_index = p0["path"], 0

            return self.active_path != before

    def get_recordings_root(self) -> str:
        with self.lock:
            if not self.active_path:
                self.recheck()
            return self.active_path or RECORDINGS_DIR_DEFAULT

    def get_all_roots(self) -> list:
        """Semua lokasi yang PERNAH/SEDANG dikonfigurasi — dipakai saat mencari file
        rekaman lama supaya tidak 'hilang' dari daftar walau sudah terjadi failover."""
        with self.lock:
            return [p["path"] for p in self.paths]

    def status(self) -> dict:
        with self.lock:
            paths_out = []
            for idx, p in enumerate(self.paths):
                h = self.check_health(p["path"])
                usage_pct = None
                if h["free_gb"] is not None and h.get("total_gb"):
                    usage_pct = round(100 - (h["free_gb"] / h["total_gb"] * 100), 1)
                paths_out.append({
                    **p, "healthy": h["healthy"], "reason": h["reason"],
                    "free_gb": h["free_gb"], "total_gb": h.get("total_gb"), "usage_pct": usage_pct,
                    "is_active": (idx == self.active_index),
                })
            return {
                "active_path": self.active_path,
                "paths": paths_out,
                "event_log": list(self.event_log)[-20:][::-1],
            }


storage_manager = StorageManager()


def _storage_monitor_loop():
    """Background thread: cek kesehatan storage tiap STORAGE_HEALTH_CHECK_INTERVAL_SEC
    detik. Kalau lokasi aktif berubah (failover/failback), restart semua proses
    rekaman yang lagi jalan supaya ffmpeg mulai nulis ke lokasi yang baru (ffmpeg
    tidak bisa pindah lokasi output di tengah proses tanpa di-restart)."""
    while True:
        try:
            changed = storage_manager.recheck()
            if changed:
                print(f"💾 [Storage] Lokasi aktif berubah -> {storage_manager.active_path}. Restart rekaman yang sedang berjalan...")
                recording_manager.relocate_all(storage_manager.get_recordings_root())
        except Exception as e:
            print(f"⚠️  Storage monitor error: {e}")
        time.sleep(STORAGE_HEALTH_CHECK_INTERVAL_SEC)


class CameraRecorder:
    """
    Merekam satu kamera LANGSUNG dari RTSP-nya lewat proses ffmpeg terpisah —
    sengaja TIDAK lewat frame yang sudah diproses AI (yang dipacu ke TARGET_STREAM_FPS
    & bisa skip frame demi hemat CPU). ffmpeg menangani decode+encode+timing sendiri
    langsung dari sumber, jadi hasil rekaman mulus mengikuti frame rate asli kamera,
    sepenuhnya independen dari beban AI/live-view.

    Direkam sebagai file ber-segmen (mis. tiap 15 menit jadi file baru) lewat muxer
    `segment` ffmpeg — supaya rekaman terus jalan tanpa batas tapi tetap jadi file-file
    berukuran wajar yang gampang dicari/diputar/dihapus per rentang waktu.
    """

    def __init__(self, cam_id: int, name: str, rtsp_url: str):
        self.cam_id = cam_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.status = {"recording": False, "error": None, "started_at": None}
        self.lock = threading.Lock()
        self._last_settings = {"resolution": "original", "crf": RECORD_CRF_DEFAULT, "segment_min": RECORD_SEGMENT_MIN_DEFAULT}

    def _out_dir(self) -> str:
        # Lokasi diambil LIVE tiap dipanggil (bukan disimpan sekali di __init__), supaya
        # kalau storage_manager baru saja failover, rekaman BERIKUTNYA (setelah restart
        # oleh _storage_monitor_loop) otomatis nulis ke lokasi yang baru.
        d = os.path.join(storage_manager.get_recordings_root(), str(self.cam_id))
        os.makedirs(d, exist_ok=True)
        return d

    def start(self, resolution: str = "original", crf: int = RECORD_CRF_DEFAULT,
              segment_min: int = RECORD_SEGMENT_MIN_DEFAULT):
        self._last_settings = {"resolution": resolution, "crf": crf, "segment_min": segment_min}
        if not FFMPEG_BIN:
            raise RuntimeError("ffmpeg tidak ditemukan di server — install ffmpeg dulu sebelum merekam.")
        with self.lock:
            if self.running:
                return  # sudah jalan, tidak perlu start dobel
            self.running = True

        pattern = os.path.join(self._out_dir(), "%Y%m%d_%H%M%S.mp4")
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-rtsp_transport", "tcp", "-timeout", "5000000",
            "-i", self.rtsp_url,
        ]
        dims = RECORD_RESOLUTIONS.get(resolution)
        if dims:
            # scale dgn flag force_original_aspect_ratio+pad supaya rasio gambar tetap
            # proporsional (tidak gepeng) walau resolusi kamera sumber tidak persis 16:9.
            w, h = dims
            cmd += ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(max(0, min(51, crf))),
            "-pix_fmt", "yuv420p",
            "-an",  # kebanyakan CCTV RTSP tidak ada audio track yang berguna; skip biar ringan
            "-f", "segment", "-segment_time", str(max(1, segment_min) * 60),
            "-reset_timestamps", "1", "-strftime", "1",
            pattern,
        ]
        self.status["error"] = None
        self.thread = threading.Thread(target=self._run_loop, args=(cmd,), daemon=True)
        self.thread.start()

    def _run_loop(self, cmd: list):
        """Loop supervisi: kalau proses ffmpeg mati sendiri (mis. RTSP putus), otomatis
        di-restart setelah jeda singkat — mirip logika reconnect di CameraWorker."""
        retry_delay = 5
        while self.running:
            try:
                self.process = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                self.status["recording"] = True
                self.status["started_at"] = datetime.now().isoformat()
                last_lines = collections.deque(maxlen=15)
                for line in self.process.stderr:
                    last_lines.append(line.decode(errors="ignore").strip())
                self.process.wait()
                if self.running:
                    tail = last_lines[-1] if last_lines else "(tidak ada output)"
                    self.status["error"] = f"ffmpeg berhenti tak terduga: {tail}"
                    print(f"⚠️  [Rekam:{self.name}] ffmpeg berhenti, mencoba reconnect dalam {retry_delay}s. {tail}")
            except Exception as e:
                self.status["error"] = str(e)
            self.status["recording"] = False
            if self.running:
                time.sleep(retry_delay)
        self.status["recording"] = False

    def stop(self):
        self.running = False
        proc = self.process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.status["recording"] = False


class RecordingManager:
    """Registry semua CameraRecorder yang aktif, dipakai endpoint start/stop/status."""

    def __init__(self):
        self.recorders: dict[int, CameraRecorder] = {}
        self.lock = threading.Lock()

    def start(self, cam_id: int, name: str, rtsp_url: str, resolution: str, crf: int, segment_min: int) -> CameraRecorder:
        with self.lock:
            rec = self.recorders.get(cam_id)
            if not rec:
                rec = CameraRecorder(cam_id, name, rtsp_url)
                self.recorders[cam_id] = rec
        rec.start(resolution=resolution, crf=crf, segment_min=segment_min)
        return rec

    def stop(self, cam_id: int):
        rec = self.recorders.get(cam_id)
        if rec:
            rec.stop()

    def get_status(self, cam_id: int) -> dict:
        rec = self.recorders.get(cam_id)
        return dict(rec.status) if rec else {"recording": False, "error": None, "started_at": None}

    def stop_all(self):
        with self.lock:
            recs = list(self.recorders.values())
        for rec in recs:
            rec.stop()

    def relocate_all(self, new_root: str):
        """Dipanggil StorageManager setelah failover/failback — restart semua
        rekaman yang SEDANG jalan supaya ffmpeg mulai menulis ke lokasi baru.
        Rekaman yang tidak sedang jalan tidak perlu apa-apa (start berikutnya
        otomatis pakai lokasi terbaru lewat CameraRecorder._out_dir)."""
        with self.lock:
            recs = list(self.recorders.items())
        for cam_id, rec in recs:
            if not rec.status.get("recording"):
                continue
            settings = dict(rec._last_settings)
            print(f"🔀 [Rekam:{rec.name}] Pindah lokasi penyimpanan -> restart rekaman di \"{new_root}\"...")
            rec.stop()
            time.sleep(0.3)
            rec.start(**settings)


recording_manager = RecordingManager()

threading.Thread(target=_storage_monitor_loop, daemon=True).start()


def _recording_retention_cleanup_loop():
    """Background thread: hapus file rekaman yang lebih tua dari record_retention_days
    milik masing-masing kamera. record_retention_days = 0 berarti simpan selamanya
    (tidak pernah dihapus otomatis) — admin yang tanggung jawab kelola disk-nya sendiri.
    Diperiksa di SEMUA lokasi storage yang dikonfigurasi (bukan cuma yang aktif saat
    ini), supaya rekaman lama di disk yang sudah di-failover-kan pun tetap kena aturan
    retensi & tidak menumpuk selamanya di disk yang sudah tidak terpakai."""
    while True:
        try:
            conn = get_conn()
            rows = conn.execute("SELECT id, record_retention_days FROM cameras").fetchall()
            conn.close()
            retention_by_cam = {row["id"]: (row["record_retention_days"] or 0) for row in rows}

            for root in storage_manager.get_all_roots():
                if not os.path.isdir(root):
                    continue  # disk sedang tidak ke-mount/lepas -> lewati, jangan sampai error
                for cam_dir_name in os.listdir(root):
                    cam_dir = os.path.join(root, cam_dir_name)
                    if not os.path.isdir(cam_dir) or not cam_dir_name.isdigit():
                        continue
                    retention_days = retention_by_cam.get(int(cam_dir_name), RECORD_RETENTION_DAYS_DEFAULT)
                    if not retention_days or retention_days <= 0:
                        continue  # simpan selamanya
                    cutoff = time.time() - (retention_days * 86400)
                    for fname in os.listdir(cam_dir):
                        fpath = os.path.join(cam_dir, fname)
                        try:
                            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                                os.remove(fpath)
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️  Recording cleanup error: {e}")
        time.sleep(3600)  # cek tiap 1 jam, cukup buat kebutuhan retensi harian


threading.Thread(target=_recording_retention_cleanup_loop, daemon=True).start()


def _capture_retention_cleanup_loop():
    """Background thread: hapus snapshot capture (.jpg + .json sidecar) yang lebih
    tua dari CAPTURE_RETENTION_DAYS_DEFAULT hari. 0 = simpan selamanya."""
    while True:
        try:
            if CAPTURE_RETENTION_DAYS_DEFAULT > 0 and os.path.isdir(CAPTURES_DIR_DEFAULT):
                cutoff = time.time() - (CAPTURE_RETENTION_DAYS_DEFAULT * 86400)
                for cam_dir_name in os.listdir(CAPTURES_DIR_DEFAULT):
                    cam_dir = os.path.join(CAPTURES_DIR_DEFAULT, cam_dir_name)
                    if not os.path.isdir(cam_dir):
                        continue
                    for fname in os.listdir(cam_dir):
                        fpath = os.path.join(cam_dir, fname)
                        try:
                            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                                os.remove(fpath)
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️  Capture cleanup error: {e}")
        time.sleep(3600)


threading.Thread(target=_capture_retention_cleanup_loop, daemon=True).start()


class CameraManager:
    """Mengelola seluruh kamera: load dari DB, start/stop worker, CRUD."""

    def __init__(self):
        self.workers: dict[int, CameraWorker] = {}
        self.lock = threading.Lock()

    def load_all_from_db(self):
        self._auto_migrate_legacy_camera()
        conn = get_conn()
        rows = conn.execute("SELECT * FROM cameras WHERE enabled = 1").fetchall()
        conn.close()
        for row in rows:
            self.add_worker(dict(row))

    def _auto_migrate_legacy_camera(self):
        """
        Migrasi otomatis dari server.py lama (single-camera, via env var):
        Jika tabel `cameras` masih KOSONG dan LEGACY_RTSP_URL diset,
        buat kamera pertama otomatis dari RTSP_URL / NX_CAMERA_ID lama.
        Hanya jalan SEKALI — begitu ada kamera apapun di DB, migrasi di-skip selamanya.
        """
        if not LEGACY_RTSP_URL:
            return

        conn = get_conn()
        existing_count = conn.execute("SELECT COUNT(*) AS c FROM cameras").fetchone()["c"]
        if existing_count > 0:
            conn.close()
            return

        conn.execute(
            "INSERT INTO cameras (name, rtsp_url, nx_camera_id, enabled, "
            "ai_face_recognition, ai_smoking_detection, ai_fire_detection, ai_behavior_detection) "
            "VALUES (?, ?, ?, 1, 0, 0, 0, 0)",
            (LEGACY_CAMERA_NAME, LEGACY_RTSP_URL, LEGACY_NX_CAMERA_ID),
        )
        conn.commit()
        conn.close()
        print(
            f"[Migrasi ✅] Kamera lama dari env var RTSP_URL otomatis ditambahkan: "
            f"'{LEGACY_CAMERA_NAME}' → {LEGACY_RTSP_URL} "
            f"(NX ID: {LEGACY_NX_CAMERA_ID or '(kosong)'})"
        )

    def add_worker(self, cam_row: dict):
        with self.lock:
            worker = CameraWorker(cam_row)
            worker.start()
            self.workers[cam_row["id"]] = worker
        return worker

    def remove_worker(self, cam_id: int):
        with self.lock:
            worker = self.workers.pop(cam_id, None)
        if worker:
            worker.stop()

    def get(self, cam_id: int) -> Optional[CameraWorker]:
        return self.workers.get(cam_id)

    def all(self) -> dict:
        return dict(self.workers)


camera_manager = CameraManager()
camera_manager.load_all_from_db()


def _auto_resume_recordings():
    """Lanjutkan rekaman otomatis saat server (re)start, untuk kamera yang sebelumnya
    diaktifkan record_enabled=1 — supaya admin tidak perlu klik ulang start rekaman
    manual tiap kali server restart/update."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM cameras WHERE enabled = 1 AND record_enabled = 1").fetchall()
    conn.close()
    for row in rows:
        try:
            recording_manager.start(
                row["id"], row["name"], row["rtsp_url"],
                resolution=row["record_resolution"] or "original",
                crf=row["record_crf"] or RECORD_CRF_DEFAULT,
                segment_min=row["record_segment_min"] or RECORD_SEGMENT_MIN_DEFAULT,
            )
            print(f"🎬 Rekaman otomatis dilanjutkan: {row['name']}")
        except Exception as e:
            print(f"⚠️  Gagal auto-resume rekaman '{row['name']}': {e}")


_auto_resume_recordings()


@app.on_event("shutdown")
def _on_shutdown():
    print("🛑 Server shutdown — menghentikan semua proses rekaman ffmpeg dengan rapi...")
    recording_manager.stop_all()

# =========================================================
# 📊 PERFORMANCE MONITOR (CPU / RAM / GPU / AI throughput)
# =========================================================
# Riwayat disimpan di memori (rolling buffer) supaya dashboard bisa
# menggambar grafik "kinerja AI" tanpa perlu database terpisah.

STATS_HISTORY_LEN = 120     # 120 sample
STATS_POLL_INTERVAL_SEC = 3  # tiap 3 detik -> lebih ringan dari sisi psutil, histori tetap ~5 menit

_stats_lock = threading.Lock()
_stats_history: collections.deque = collections.deque(maxlen=STATS_HISTORY_LEN)
_proc = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None


def _collect_system_snapshot() -> dict:
    """Ambil satu snapshot kinerja sistem + AI saat ini."""
    ts = time.time()

    if PSUTIL_AVAILABLE:
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        proc_cpu = _proc.cpu_percent(interval=None) if _proc else 0.0
        try:
            proc_mem_mb = round(_proc.memory_info().rss / (1024 * 1024), 1) if _proc else 0.0
        except Exception:
            proc_mem_mb = 0.0
        try:
            load1, load5, load15 = os.getloadavg()
        except (OSError, AttributeError):
            load1 = load5 = load15 = 0.0
        disk = psutil.disk_usage("/")
    else:
        cpu_total, cpu_per_core, proc_cpu, proc_mem_mb = 0.0, [], 0.0, 0.0
        load1 = load5 = load15 = 0.0
        mem = type("M", (), {"percent": 0.0, "used": 0, "total": 0})()
        disk = type("D", (), {"percent": 0.0, "used": 0, "total": 0})()

    gpus = []
    jetson_info = None
    if IS_JETSON and JTOP_AVAILABLE:
        # Jetson: baca via jtop (butuh service jtop.service jalan di background, lihat
        # scripts/setup_jetson_orin_nx.sh). Dibuka-tutup tiap snapshot (bukan disimpan
        # sebagai koneksi global) supaya tidak memblokir kalau service jtop restart.
        try:
            with jtop(interval=0.1) as jt:
                if jt.ok():
                    gpu_stats = jt.gpu
                    # jtop mengembalikan dict per-GPU (Jetson biasanya cuma 1: 'gpu' / 'ga10b' dst)
                    first_gpu = next(iter(gpu_stats.values())) if gpu_stats else {}
                    gpu_load = first_gpu.get("status", {}).get("load", 0.0)
                    gpus.append({
                        "name": JETSON_MODEL or "Jetson GPU",
                        "load_pct": round(gpu_load, 1),
                        "mem_used_mb": round(jt.memory.get("RAM", {}).get("used", 0) / 1024, 1),
                        "mem_total_mb": round(jt.memory.get("RAM", {}).get("tot", 0) / 1024, 1),
                        "temp_c": jt.temperature.get("gpu", jt.temperature.get("GPU", {})).get("temp"),
                    })
                    jetson_info = {
                        "model": JETSON_MODEL,
                        "power_mode": jt.nvpmodel.name if jt.nvpmodel else None,
                        "power_draw_w": round(jt.power.get("tot", {}).get("power", 0) / 1000, 1) if jt.power else None,
                        "jetson_clocks": bool(jt.jetson_clocks) if jt.jetson_clocks is not None else None,
                    }
        except Exception as e:
            jetson_info = {"error": f"jtop tidak bisa dibaca: {e}"}
    elif GPUTIL_AVAILABLE:
        # Jalur non-Jetson (PC/server dengan GPU NVIDIA desktop, punya nvidia-smi)
        try:
            for g in GPUtil.getGPUs():
                gpus.append({
                    "name": g.name,
                    "load_pct": round(g.load * 100, 1),
                    "mem_used_mb": round(g.memoryUsed, 1),
                    "mem_total_mb": round(g.memoryTotal, 1),
                    "temp_c": g.temperature,
                })
        except Exception:
            pass

    # --- Per-kamera: FPS live + status AI ---
    cams_perf = []
    total_fps = 0.0
    active_ai_cams = 0
    for cam_id, worker in camera_manager.all().items():
        ai = worker.ai_settings
        ai_active = any(ai.values())
        if ai_active:
            active_ai_cams += 1
        total_fps += worker.perf.get("fps", 0.0)
        cams_perf.append({
            "id": cam_id,
            "name": worker.name,
            "fps": worker.perf.get("fps", 0.0),
            "yolo_ms_avg": worker.perf.get("yolo_ms_avg", 0.0),
            "ai_active": ai_active,
            "open": worker.status.get("open", False),
        })

    snapshot = {
        "ts": ts,
        "cpu": {
            "total_pct": cpu_total,
            "per_core_pct": cpu_per_core,
            "cores": len(cpu_per_core) if cpu_per_core else (os.cpu_count() or 0),
            "load_avg": {"1m": load1, "5m": load5, "15m": load15},
        },
        "memory": {
            "pct": mem.percent,
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "total_gb": round(mem.total / (1024 ** 3), 2),
        },
        "disk": {
            "pct": disk.percent,
            "used_gb": round(disk.used / (1024 ** 3), 2),
            "total_gb": round(disk.total / (1024 ** 3), 2),
        },
        "process": {
            "cpu_pct": proc_cpu,
            "mem_mb": proc_mem_mb,
            "threads": threading.active_count(),
        },
        "gpu": gpus,
        "jetson": jetson_info,   # None kalau bukan Jetson / jtop tidak tersedia
        "ai": {
            "total_cameras": len(cams_perf),
            "active_ai_cameras": active_ai_cams,
            "total_fps": round(total_fps, 1),
            "yolo_model": YOLO_MODEL_PATH,
            "yolo_device": "GPU (cuda:0)" if CUDA_AVAILABLE else "CPU",
            "yolo_every_n_frames": YOLO_EVERY_N_FRAMES,
            "yolo_img_size": YOLO_IMG_SIZE,
            "cameras": cams_perf,
        },
    }
    return snapshot


def _stats_poller():
    """Loop background: ambil snapshot tiap STATS_POLL_INTERVAL_SEC dan simpan ke histori."""
    time.sleep(2)
    if PSUTIL_AVAILABLE:
        psutil.cpu_percent(interval=None)  # panggilan pertama psutil selalu 0.0, warm-up
        if _proc:
            _proc.cpu_percent(interval=None)
    while True:
        try:
            snap = _collect_system_snapshot()
            with _stats_lock:
                _stats_history.append(snap)
        except Exception as e:
            print(f"[Stats Poller ERR] {e}")
        time.sleep(STATS_POLL_INTERVAL_SEC)


threading.Thread(target=_stats_poller, daemon=True).start()

# =========================================================
# 📦 PYDANTIC MODELS
# =========================================================

class CameraCreate(BaseModel):
    name: str
    rtsp_url: str
    nx_camera_id: str = ""
    brand: str = ""  # info saja (hikvision/dahua/uniview/axis/generic), tidak divalidasi/disimpan khusus


# --- 📡 ONVIF (auto-discovery, auto-config RTSP, PTZ multi-brand) ---

class OnvifDiscoverRequest(BaseModel):
    timeout: float = 4.0  # detik menunggu balasan WS-Discovery di jaringan lokal


class OnvifProbeRequest(BaseModel):
    """Konek langsung ke 1 kamera by IP (dari hasil discovery ATAU diisi manual
    kalau kamera tidak ketemu via broadcast, mis. beda subnet/VLAN)."""
    host: str
    port: int = 80
    username: str = ""
    password: str = ""


class OnvifCameraCreate(BaseModel):
    """Simpan kamera hasil probe ONVIF: RTSP diambil OTOMATIS dari profile yang
    dipilih user di dashboard (tidak perlu ketik manual rtsp://...)."""
    name: str
    nx_camera_id: str = ""
    brand: str = ""
    host: str
    port: int = 80
    username: str = ""
    password: str = ""
    profile_token: str            # dipilih dari hasil /api/onvif/probe
    rtsp_url: str                 # ikut dikirim dari hasil probe (sudah termasuk kredensial)
    ptz_supported: bool = False


class PTZMoveRequest(BaseModel):
    """Gerakan PTZ 'continuous' ala joystick — kamera terus bergerak sampai
    endpoint /stop dipanggil. pan/tilt/zoom masing2 -1.0..1.0."""
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0


class PTZPresetSave(BaseModel):
    name: str
    preset_token: Optional[str] = None  # isi kalau mau TIMPA preset lama, kosongkan utk buat baru


class CameraAIToggle(BaseModel):
    face_recognition: Optional[bool] = None
    smoking_detection: Optional[bool] = None
    fire_detection: Optional[bool] = None
    behavior_detection: Optional[bool] = None
    people_counting: Optional[bool] = None
    vehicle_counting: Optional[bool] = None
    animal_counting: Optional[bool] = None
    parking_detection: Optional[bool] = None


class CountingLineUpdate(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class CountingConfigUpdate(BaseModel):
    """
    Konfigurasi lengkap area/garis counting untuk 1 kamera:
    - shape: "line" (garis virtual 2 titik) atau "polygon" (area bebas, N titik)
    - direction: "both" (masuk & keluar dihitung), "in" (masuk saja), "out" (keluar saja)
    - line: dipakai kalau shape="line"
    - polygon: list titik [[x,y], ...] (min 3), dipakai kalau shape="polygon"
    - line_style: "rope" (gaya tali/dipilin), "dashed", atau "solid"
    - line_thickness: ketebalan garis/tepi area, 1-6 (bisa diperkecil)
    """
    shape: Optional[str] = None
    direction: Optional[str] = None
    line: Optional[CountingLineUpdate] = None
    polygon: Optional[list] = None
    line_style: Optional[str] = None
    line_thickness: Optional[float] = None


class RecordingSettingsUpdate(BaseModel):
    """
    Setting rekaman 1 kamera:
    - resolution: "original", "1080p", "720p", "480p", "360p" — makin kecil = file makin hemat
    - crf: 0-51 (standar x264), makin BESAR = makin terkompres/kecil filenya tapi kualitas turun.
      18-23 = kualitas bagus, 24-28 = hemat disk & masih layak buat CCTV, 29+ = sangat hemat.
    - segment_min: durasi tiap file rekaman (menit) sebelum pindah ke file baru
    - retention_days: hapus otomatis rekaman lebih tua dari sekian hari (0 = simpan selamanya)
    """
    resolution: Optional[str] = None
    crf: Optional[int] = None
    segment_min: Optional[int] = None
    retention_days: Optional[int] = None


class StoragePathItem(BaseModel):
    path: str
    label: Optional[str] = ""


class StoragePathsUpdate(BaseModel):
    """Daftar lokasi penyimpanan LENGKAP dgn urutan prioritas — index ke-0 = primary,
    index selanjutnya = cadangan failover berurutan. Kirim seluruh daftar tiap update
    (bukan tambah satu-satu) supaya urutan prioritas gampang diatur ulang dari dashboard."""
    paths: list[StoragePathItem]


class StoragePathTest(BaseModel):
    path: str


class ParkingSlotCreate(BaseModel):
    """1 slot parkir baru: label bebas (mis. 'A1') + poligon custom (min 3 titik,
    tiap titik [x,y] dinormalisasi 0-1 relatif terhadap frame kamera)."""
    label: str = ""
    polygon: list


class ParkingSlotUpdate(BaseModel):
    label: Optional[str] = None
    polygon: Optional[list] = None


class GlobalAISettingsUpdate(BaseModel):
    yolo_every_n_frames: Optional[int] = None       # jalankan YOLO tiap N frame (makin besar = makin ringan CPU)
    recognition_confidence_threshold: Optional[int] = None  # 0-100, makin kecil makin ketat/ akurat
    face_every_n_frames: Optional[int] = None        # jalankan face recognition tiap N frame (makin besar = makin ringan CPU)
    face_accept_sim: Optional[float] = None          # 0-1, ambang skor minimal supaya sebuah nama diterima
    face_accept_margin: Optional[float] = None       # 0-1, jarak minimal kandidat 1 vs kandidat orang lain
    target_stream_fps: Optional[float] = None        # patok FPS stream per kamera (mis. 15 utk CPU lemah)

    # --- 🔥 Fire / Smoke ---
    fire_sensitivity: Optional[int] = None           # 1 (longgar) - 10 (ketat warnanya)
    fire_min_area_pct: Optional[float] = None        # % luas frame minimum supaya tidak noise
    fire_confirm_frames: Optional[int] = None        # siklus berturut sebelum alert ditembak
    fire_every_n_frames: Optional[int] = None        # cadence deteksi warna api/asap

    # --- 🚬 Smoking ---
    smoking_head_zone_pct: Optional[float] = None    # % atas box orang yg dianggap zona tangan-ke-mulut
    smoking_confirm_frames: Optional[int] = None

    # --- 🚨 Fall / Behavior ---
    behavior_confirm_frames: Optional[int] = None


# =========================================================
# 🌐 DASHBOARD
# =========================================================

_PWA_HEAD_INJECT = """
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#0f172a">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="VMS">
<link rel="apple-touch-icon" href="/static/icons/icon-192.png">
<link rel="icon" type="image/png" href="data:image/png;base64,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">
<script>
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((e) => console.warn("SW register failed:", e));
  });
}
</script>
"""

def _inject_pwa_tags(html: str) -> str:
    lower = html.lower()
    idx = lower.find("<head>")
    if idx != -1:
        insert_at = idx + len("<head>")
        return html[:insert_at] + _PWA_HEAD_INJECT + html[insert_at:]
    # Fallback kalau tag <head> tidak ditemukan persis (mis. beda kapitalisasi/atribut)
    return _PWA_HEAD_INJECT + html


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return _inject_pwa_tags(html)


# =========================================================
# 📹 CAMERA CRUD (Dynamic add/remove)
# =========================================================

@app.get("/api/cameras")
async def list_cameras():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM cameras ORDER BY id ASC").fetchall()
    conn.close()
    result = []
    for row in rows:
        cam = dict(row)
        cam.pop("onvif_password", None)  # jangan bocorkan sandi ONVIF ke response API
        worker = camera_manager.get(cam["id"])
        cam["live_status"] = worker.status if worker else {"open": False, "error": "Worker tidak aktif"}
        result.append(cam)
    return {"cameras": result, "total": len(result)}


@app.post("/api/cameras/test_rtsp")
async def test_rtsp(cam: CameraCreate):
    """
    Tes koneksi RTSP TANPA menyimpan ke database.
    Dipakai dashboard untuk validasi URL sebelum klik 'Simpan'.
    """
    def _test():
        cap = cv2.VideoCapture(cam.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        opened = cap.isOpened()
        frame = None
        if opened:
            ret, frame = cap.read()
            opened = ret
        cap.release()
        return opened, frame

    loop = asyncio.get_event_loop()
    ok, frame = await loop.run_in_executor(None, _test)

    if not ok:
        return JSONResponse(status_code=422, content={
            "status": "error",
            "detail": "Tidak bisa konek ke RTSP. Cek IP, port, username/password, atau path channel.",
        })

    _, buf = cv2.imencode(".jpg", cv2.resize(frame, (320, 180)))
    preview = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return {"status": "ok", "preview": preview}


@app.post("/api/cameras")
async def add_camera(cam: CameraCreate):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO cameras (name, rtsp_url, nx_camera_id) VALUES (?, ?, ?)",
        (cam.name, cam.rtsp_url, cam.nx_camera_id),
    )
    conn.commit()
    new_id = cur.lastrowid
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (new_id,)).fetchone()
    conn.close()

    camera_manager.add_worker(dict(row))
    return {"status": "ok", "camera": dict(row)}


# =========================================================
# 📡 ONVIF — DISCOVERY, PROBE, & TAMBAH KAMERA OTOMATIS
# =========================================================
# Alur di dashboard: (1) Cari kamera di jaringan [opsional, kalau 1 subnet] ATAU
# isi IP manual -> (2) Hubungkan & Ambil Profil (probe) dengan user/pass ONVIF
# kamera -> (3) pilih stream (main/sub) dari daftar profile -> (4) Simpan kamera.
# RTSP URL TIDAK perlu diketik manual — diambil otomatis dari kamera via ONVIF
# Media Service, dan sudah termasuk info apakah kamera itu punya PTZ atau tidak.

@app.post("/api/onvif/discover")
async def onvif_discover(req: OnvifDiscoverRequest):
    """Broadcast WS-Discovery ke jaringan lokal server ini (bukan dari browser
    user!) — jadi hanya menemukan kamera yang satu subnet/VLAN dengan server
    VMS. Kalau kamera ada di subnet lain, gunakan /api/onvif/probe dengan IP
    yang diisi manual."""
    loop = asyncio.get_event_loop()
    try:
        devices = await loop.run_in_executor(None, onvif_ptz.discover_onvif_devices, req.timeout)
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
    return {"status": "ok", "devices": devices, "total": len(devices)}


@app.post("/api/onvif/probe")
async def onvif_probe(req: OnvifProbeRequest):
    """Konek ke 1 kamera pakai kredensial ONVIF-nya, ambil info device + semua
    media profile (stream) beserta RTSP URL siap-pakai + status dukungan PTZ.
    Tidak menyimpan apa pun ke database — cuma untuk ditampilkan dashboard
    sebelum user klik 'Simpan kamera'."""
    def _do():
        return onvif_ptz.probe_onvif_camera(req.host, req.port, req.username, req.password)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _do)
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=422, content={
            "status": "error",
            "detail": f"Gagal konek ke kamera ONVIF. Cek IP/port/username/password. ({e})",
        })

    return {
        "status": "ok",
        "manufacturer": result.manufacturer,
        "model": result.model,
        "firmware": result.firmware,
        "serial": result.serial,
        "ptz_supported": result.ptz_supported,
        "profiles": [
            {
                "token": p.token, "name": p.name, "rtsp_url": p.rtsp_url,
                "ptz_supported": p.ptz_supported, "resolution": p.resolution,
            }
            for p in result.profiles
        ],
    }


@app.post("/api/cameras/onvif")
async def add_camera_onvif(cam: OnvifCameraCreate):
    """Simpan kamera baru dari hasil /api/onvif/probe. RTSP URL & info PTZ
    sudah dikirim dari dashboard (hasil probe), jadi di sini tinggal ditulis
    ke tabel `cameras` yang sama dengan kamera RTSP manual — worker live-view,
    AI, rekaman dsb otomatis jalan seperti kamera biasa. Field onvif_* dipakai
    khusus untuk kontrol PTZ (lihat endpoint /api/cameras/{id}/ptz/*)."""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO cameras (name, rtsp_url, nx_camera_id, camera_brand, "
        "onvif_host, onvif_port, onvif_username, onvif_password, "
        "onvif_profile_token, ptz_supported) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (cam.name, cam.rtsp_url, cam.nx_camera_id, cam.brand,
         cam.host, cam.port, cam.username, cam.password,
         cam.profile_token, 1 if cam.ptz_supported else 0),
    )
    conn.commit()
    new_id = cur.lastrowid
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (new_id,)).fetchone()
    conn.close()

    camera_manager.add_worker(dict(row))
    result = dict(row)
    result.pop("onvif_password", None)
    return {"status": "ok", "camera": result}


# =========================================================
# 🎮 KONTROL PTZ (Pan-Tilt-Zoom) — MULTI-BRAND via ONVIF
# =========================================================
# Berlaku untuk kamera apa pun yang disimpan lewat /api/cameras/onvif dengan
# ptz_supported=1 (Hikvision, Dahua, Uniview, Axis, Bosch, Hanwha, TP-Link/VIGI,
# Reolink, dll — semua yang comply ke ONVIF Profile S/T PTZ Service, TIDAK perlu
# SDK/plugin khusus per merk).

def _get_ptz_camera_row(cam_id: int) -> sqlite3.Row:
    conn = get_conn()
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan")
    if not row["ptz_supported"] or not row["onvif_host"]:
        raise HTTPException(
            status_code=422,
            detail="Kamera ini tidak dikonfigurasi sebagai kamera ONVIF ber-PTZ. "
                   "Tambahkan ulang lewat menu 'Tambah kamera → ONVIF' agar PTZ aktif.",
        )
    return row


def _get_ptz_controller(cam_id: int) -> onvif_ptz.OnvifPTZController:
    row = _get_ptz_camera_row(cam_id)
    return onvif_ptz.ptz_registry.get_or_create(
        cam_id, row["onvif_host"], row["onvif_port"] or 80,
        row["onvif_username"] or "", row["onvif_password"] or "",
        row["onvif_profile_token"] or None,
    )


@app.post("/api/cameras/{cam_id}/ptz/move")
async def ptz_move(cam_id: int, req: PTZMoveRequest):
    """Mulai gerak PTZ ke arah tertentu (continuous — terus bergerak sampai
    /ptz/stop dipanggil). Dashboard memanggil ini saat tombol arah DITEKAN,
    dan /stop saat tombol DILEPAS — persis joystick fisik di NVR."""
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, ctrl.continuous_move, req.pan, req.tilt, req.zoom)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Perintah PTZ gagal: {e}"})
    return {"status": "ok"}


@app.post("/api/cameras/{cam_id}/ptz/stop")
async def ptz_stop(cam_id: int):
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, ctrl.stop)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Perintah stop PTZ gagal: {e}"})
    return {"status": "ok"}


@app.get("/api/cameras/{cam_id}/ptz/status")
async def ptz_status(cam_id: int):
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        status = await loop.run_in_executor(None, ctrl.get_status)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Gagal ambil status PTZ: {e}"})
    return {"status": "ok", **status}


@app.post("/api/cameras/{cam_id}/ptz/home")
async def ptz_home(cam_id: int):
    """Gerak ke posisi 'home' yang tersimpan di kamera (kalau ada)."""
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, ctrl.goto_home)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Gagal ke posisi home: {e}"})
    return {"status": "ok"}


@app.post("/api/cameras/{cam_id}/ptz/home/set")
async def ptz_set_home(cam_id: int):
    """Simpan posisi PTZ SAAT INI sebagai posisi 'home' baru di kamera."""
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, ctrl.set_home)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Gagal simpan posisi home: {e}"})
    return {"status": "ok"}


@app.get("/api/cameras/{cam_id}/ptz/presets")
async def ptz_list_presets(cam_id: int):
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        presets = await loop.run_in_executor(None, ctrl.get_presets)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Gagal ambil daftar preset: {e}"})
    return {"status": "ok", "presets": presets, "total": len(presets)}


@app.post("/api/cameras/{cam_id}/ptz/presets")
async def ptz_save_preset(cam_id: int, body: PTZPresetSave):
    """Simpan posisi PTZ saat ini sebagai preset baru (atau timpa preset lama
    kalau preset_token diisi)."""
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        token = await loop.run_in_executor(None, ctrl.set_preset, body.name, body.preset_token)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Gagal simpan preset: {e}"})
    return {"status": "ok", "preset_token": token}


@app.post("/api/cameras/{cam_id}/ptz/presets/{preset_token}/goto")
async def ptz_goto_preset(cam_id: int, preset_token: str):
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, ctrl.goto_preset, preset_token)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Gagal pindah ke preset: {e}"})
    return {"status": "ok"}


@app.delete("/api/cameras/{cam_id}/ptz/presets/{preset_token}")
async def ptz_delete_preset(cam_id: int, preset_token: str):
    ctrl = _get_ptz_controller(cam_id)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, ctrl.remove_preset, preset_token)
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "error", "detail": f"Gagal hapus preset: {e}"})
    return {"status": "ok"}


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan")
    conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
    conn.commit()
    conn.close()

    camera_manager.remove_worker(cam_id)
    recording_manager.stop(cam_id)  # file rekaman lama sengaja TIDAK dihapus, tetap tersimpan sbg arsip
    onvif_ptz.ptz_registry.drop(cam_id)
    return {"status": "ok"}


@app.post("/api/cameras/{cam_id}/toggle_enabled")
async def toggle_camera_enabled(cam_id: int, enabled: bool):
    conn = get_conn()
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan")
    conn.execute("UPDATE cameras SET enabled=? WHERE id=?", (1 if enabled else 0, cam_id))
    conn.commit()
    conn.close()

    if enabled:
        row2 = get_conn().execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
        camera_manager.add_worker(dict(row2))
        if row2["record_enabled"]:
            try:
                recording_manager.start(
                    cam_id, row2["name"], row2["rtsp_url"],
                    resolution=row2["record_resolution"] or "original",
                    crf=row2["record_crf"] or RECORD_CRF_DEFAULT,
                    segment_min=row2["record_segment_min"] or RECORD_SEGMENT_MIN_DEFAULT,
                )
            except Exception:
                pass
    else:
        camera_manager.remove_worker(cam_id)
        recording_manager.stop(cam_id)
    return {"status": "ok", "enabled": enabled}


@app.get("/api/settings/ai")
async def get_ai_settings():
    """Setting AI global (berlaku untuk semua kamera), dipakai menu Settings di dashboard."""
    return {
        "yolo_every_n_frames": YOLO_EVERY_N_FRAMES,
        "face_every_n_frames": FACE_EVERY_N_FRAMES,
        "recognition_confidence_threshold": RECOGNITION_CONFIDENCE_THRESHOLD,  # warisan, tidak dipakai
        "face_accept_sim": face_engine.status()["accept_sim"],
        "face_accept_margin": face_engine.status()["accept_margin"],
        "target_stream_fps": TARGET_STREAM_FPS,
        "yolo_img_size": YOLO_IMG_SIZE,
        "smoking_proxy_classes": sorted(SMOKING_PROXY_CLASSES),
        "yolo_model": "yolov8n",
        "max_concurrent_ai_threads": MAX_CONCURRENT_AI_THREADS,
        "cpu_cores": CPU_CORES,
        "low_power_mode": LOW_POWER_MODE,
        # --- fire/smoke ---
        "fire_sensitivity": FIRE_SENSITIVITY,
        "fire_min_area_pct": FIRE_MIN_AREA_PCT,
        "fire_confirm_frames": FIRE_CONFIRM_FRAMES,
        "fire_every_n_frames": FIRE_EVERY_N_FRAMES,
        # --- smoking ---
        "smoking_head_zone_pct": SMOKING_HEAD_ZONE_PCT,
        "smoking_confirm_frames": SMOKING_CONFIRM_FRAMES,
        # --- behavior/fall ---
        "behavior_confirm_frames": BEHAVIOR_CONFIRM_FRAMES,
    }


@app.post("/api/settings/ai")
async def update_ai_settings(settings: GlobalAISettingsUpdate):
    """
    Update setting AI global secara live (tanpa restart server).
    - yolo_every_n_frames lebih besar -> YOLO jalan lebih jarang -> lebih ringan CPU, deteksi lebih lambat.
    - recognition_confidence_threshold lebih kecil -> pengenalan wajah lebih ketat (harus lebih mirip).
    - fire_sensitivity lebih besar -> rentang warna dilonggarkan -> lebih sensitif tapi makin gampang false-positive.
    - *_confirm_frames lebih besar -> lebih tahan false-positif tapi alert lebih lambat muncul.
    """
    global YOLO_EVERY_N_FRAMES, RECOGNITION_CONFIDENCE_THRESHOLD, FACE_EVERY_N_FRAMES, TARGET_STREAM_FPS
    global FIRE_SENSITIVITY, FIRE_MIN_AREA_PCT, FIRE_CONFIRM_FRAMES, FIRE_EVERY_N_FRAMES
    global SMOKING_HEAD_ZONE_PCT, SMOKING_CONFIRM_FRAMES, BEHAVIOR_CONFIRM_FRAMES

    if settings.yolo_every_n_frames is not None:
        if settings.yolo_every_n_frames < 1 or settings.yolo_every_n_frames > 60:
            raise HTTPException(status_code=422, detail="yolo_every_n_frames harus antara 1-60")
        YOLO_EVERY_N_FRAMES = settings.yolo_every_n_frames

    if settings.face_every_n_frames is not None:
        if settings.face_every_n_frames < 1 or settings.face_every_n_frames > 60:
            raise HTTPException(status_code=422, detail="face_every_n_frames harus antara 1-60")
        FACE_EVERY_N_FRAMES = settings.face_every_n_frames

    if settings.recognition_confidence_threshold is not None:
        # WARISAN: dulu ini ambang jarak LBPH mentah. Mesin wajah yang baru
        # menilai secara relatif terhadap sebaran jarak ke orang lain, sehingga
        # tidak ada lagi angka mutlak yang bermakna di sini. Nilainya tetap
        # diterima & disimpan supaya dashboard lama tidak error, tapi TIDAK
        # dipakai. Gunakan face_accept_sim / face_accept_margin.
        if settings.recognition_confidence_threshold < 0 or settings.recognition_confidence_threshold > 150:
            raise HTTPException(status_code=422, detail="recognition_confidence_threshold harus antara 0-150")
        RECOGNITION_CONFIDENCE_THRESHOLD = settings.recognition_confidence_threshold

    if settings.face_accept_sim is not None:
        if not (0.0 <= settings.face_accept_sim <= 1.0):
            raise HTTPException(status_code=422, detail="face_accept_sim harus antara 0.0-1.0")
        import enpidix_face_engine as _fe
        _fe.ACCEPT_SIM = float(settings.face_accept_sim)

    if settings.face_accept_margin is not None:
        if not (0.0 <= settings.face_accept_margin <= 1.0):
            raise HTTPException(status_code=422, detail="face_accept_margin harus antara 0.0-1.0")
        import enpidix_face_engine as _fe
        _fe.ACCEPT_MARGIN = float(settings.face_accept_margin)

    if settings.target_stream_fps is not None:
        if settings.target_stream_fps < 1 or settings.target_stream_fps > 30:
            raise HTTPException(status_code=422, detail="target_stream_fps harus antara 1-30")
        TARGET_STREAM_FPS = settings.target_stream_fps

    if settings.fire_sensitivity is not None:
        if settings.fire_sensitivity < 1 or settings.fire_sensitivity > 10:
            raise HTTPException(status_code=422, detail="fire_sensitivity harus antara 1-10")
        FIRE_SENSITIVITY = settings.fire_sensitivity

    if settings.fire_min_area_pct is not None:
        if settings.fire_min_area_pct < 0.05 or settings.fire_min_area_pct > 10:
            raise HTTPException(status_code=422, detail="fire_min_area_pct harus antara 0.05-10")
        FIRE_MIN_AREA_PCT = settings.fire_min_area_pct

    if settings.fire_confirm_frames is not None:
        if settings.fire_confirm_frames < 1 or settings.fire_confirm_frames > 20:
            raise HTTPException(status_code=422, detail="fire_confirm_frames harus antara 1-20")
        FIRE_CONFIRM_FRAMES = settings.fire_confirm_frames

    if settings.fire_every_n_frames is not None:
        if settings.fire_every_n_frames < 1 or settings.fire_every_n_frames > 60:
            raise HTTPException(status_code=422, detail="fire_every_n_frames harus antara 1-60")
        FIRE_EVERY_N_FRAMES = settings.fire_every_n_frames

    if settings.smoking_head_zone_pct is not None:
        if settings.smoking_head_zone_pct < 10 or settings.smoking_head_zone_pct > 80:
            raise HTTPException(status_code=422, detail="smoking_head_zone_pct harus antara 10-80")
        SMOKING_HEAD_ZONE_PCT = settings.smoking_head_zone_pct

    if settings.smoking_confirm_frames is not None:
        if settings.smoking_confirm_frames < 1 or settings.smoking_confirm_frames > 20:
            raise HTTPException(status_code=422, detail="smoking_confirm_frames harus antara 1-20")
        SMOKING_CONFIRM_FRAMES = settings.smoking_confirm_frames

    if settings.behavior_confirm_frames is not None:
        if settings.behavior_confirm_frames < 1 or settings.behavior_confirm_frames > 20:
            raise HTTPException(status_code=422, detail="behavior_confirm_frames harus antara 1-20")
        BEHAVIOR_CONFIRM_FRAMES = settings.behavior_confirm_frames

    return {
        "status": "ok",
        "yolo_every_n_frames": YOLO_EVERY_N_FRAMES,
        "face_every_n_frames": FACE_EVERY_N_FRAMES,
        "recognition_confidence_threshold": RECOGNITION_CONFIDENCE_THRESHOLD,  # warisan, tidak dipakai
        "face_accept_sim": face_engine.status()["accept_sim"],
        "face_accept_margin": face_engine.status()["accept_margin"],
        "target_stream_fps": TARGET_STREAM_FPS,
        "fire_sensitivity": FIRE_SENSITIVITY,
        "fire_min_area_pct": FIRE_MIN_AREA_PCT,
        "fire_confirm_frames": FIRE_CONFIRM_FRAMES,
        "fire_every_n_frames": FIRE_EVERY_N_FRAMES,
        "smoking_head_zone_pct": SMOKING_HEAD_ZONE_PCT,
        "smoking_confirm_frames": SMOKING_CONFIRM_FRAMES,
        "behavior_confirm_frames": BEHAVIOR_CONFIRM_FRAMES,
    }


@app.post("/api/cameras/{cam_id}/ai_toggle")
async def toggle_camera_ai(cam_id: int, toggle: CameraAIToggle):
    """Toggle AI feature per-kamera. Hanya field yang dikirim yang berubah."""
    _camera_row_or_404(cam_id)
    worker = camera_manager.get(cam_id)

    updates = {k: v for k, v in toggle.dict().items() if v is not None}
    if worker:
        worker.update_ai_settings(**updates)

    # Persist ke DB juga
    conn = get_conn()
    set_clauses = []
    values = []
    col_map = {
        "face_recognition": "ai_face_recognition",
        "smoking_detection": "ai_smoking_detection",
        "fire_detection": "ai_fire_detection",
        "behavior_detection": "ai_behavior_detection",
        "people_counting": "ai_people_counting",
        "vehicle_counting": "ai_vehicle_counting",
        "animal_counting": "ai_animal_counting",
        "parking_detection": "ai_parking_detection",
    }
    for k, v in updates.items():
        set_clauses.append(f"{col_map[k]}=?")
        values.append(1 if v else 0)
    if set_clauses:
        values.append(cam_id)
        conn.execute(f"UPDATE cameras SET {', '.join(set_clauses)} WHERE id=?", values)
        conn.commit()
    conn.close()

    if worker:
        return {"status": "ok", "ai_settings": worker.ai_settings}
    row = _camera_row_or_404(cam_id)
    return {"status": "ok", "camera_active": False, "ai_settings": {
        k: bool(row.get(col)) for k, col in col_map.items()
    }}


# =========================================================
# 📊 PEOPLE / VEHICLE COUNTING (virtual line)
# =========================================================

@app.get("/api/cameras/{cam_id}/counting/line")
async def get_counting_line(cam_id: int):
    """Ambil koordinat garis virtual (0-1, relatif ke frame) untuk kamera ini."""
    worker = camera_manager.get(cam_id)
    if worker:
        x1, y1, x2, y2 = worker.count_line
    else:
        line = _counting_config_from_row(_camera_row_or_404(cam_id))["line"]
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    return {"camera_id": cam_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


@app.post("/api/cameras/{cam_id}/counting/line")
async def set_counting_line(cam_id: int, line: CountingLineUpdate):
    """
    Ubah posisi garis virtual counting (adjustable, dipakai dashboard buat drag
    titik ujung garis di atas preview kamera). Koordinat 0-1 relatif terhadap
    lebar/tinggi frame, supaya tetap valid di resolusi berapa pun.
    """
    _camera_row_or_404(cam_id)
    worker = camera_manager.get(cam_id)
    for v, label in [(line.x1, "x1"), (line.y1, "y1"), (line.x2, "x2"), (line.y2, "y2")]:
        if not (0.0 <= v <= 1.0):
            raise HTTPException(status_code=422, detail=f"Koordinat {label} harus di rentang 0-1.")

    if worker:
        worker.set_count_line(line.x1, line.y1, line.x2, line.y2)
    conn = get_conn()
    conn.execute(
        "UPDATE cameras SET count_line_x1=?, count_line_y1=?, count_line_x2=?, count_line_y2=? WHERE id=?",
        (line.x1, line.y1, line.x2, line.y2, cam_id),
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "line": {"x1": line.x1, "y1": line.y1, "x2": line.x2, "y2": line.y2}}


@app.get("/api/cameras/{cam_id}/counting/config")
async def get_counting_config(cam_id: int):
    """
    Ambil konfigurasi lengkap counting kamera ini: bentuk (garis/area), arah yang
    dihitung, titik garis/poligon, dan gaya tampilan garis (tali/putus-putus/solid).
    """
    worker = camera_manager.get(cam_id)
    if worker:
        return {"camera_id": cam_id, "camera_active": True, **worker.get_counting_config()}
    row = _camera_row_or_404(cam_id)
    return {"camera_id": cam_id, "camera_active": False, **_counting_config_from_row(row)}


@app.post("/api/cameras/{cam_id}/counting/config")
async def set_counting_config(cam_id: int, cfg: CountingConfigUpdate):
    """
    Simpan konfigurasi counting kamera ini sekaligus (dipakai editor garis/area di
    dashboard): bentuk area (garis 2 titik ATAU poligon N titik yang bisa
    disesuaikan bentuknya), arah yang dihitung (masuk saja/keluar saja/dua-duanya),
    dan gaya tampilan garis (gaya tali yang bisa diperkecil, putus-putus, atau solid).
    """
    _camera_row_or_404(cam_id)
    worker = camera_manager.get(cam_id)

    conn = get_conn()
    set_clauses, values = [], []

    if cfg.shape is not None:
        if cfg.shape not in ("line", "polygon"):
            raise HTTPException(status_code=422, detail="shape harus 'line' atau 'polygon'.")
        if worker:
            worker.set_count_shape(cfg.shape)
        set_clauses.append("count_shape=?"); values.append(cfg.shape)

    if cfg.direction is not None:
        if cfg.direction not in ("both", "in", "out"):
            raise HTTPException(status_code=422, detail="direction harus 'both', 'in', atau 'out'.")
        if worker:
            worker.set_count_direction(cfg.direction)
        set_clauses.append("count_direction=?"); values.append(cfg.direction)

    if cfg.line is not None:
        line = cfg.line
        for v, label in [(line.x1, "x1"), (line.y1, "y1"), (line.x2, "x2"), (line.y2, "y2")]:
            if not (0.0 <= v <= 1.0):
                raise HTTPException(status_code=422, detail=f"Koordinat garis {label} harus di rentang 0-1.")
        if worker:
            worker.set_count_line(line.x1, line.y1, line.x2, line.y2)
        set_clauses += ["count_line_x1=?", "count_line_y1=?", "count_line_x2=?", "count_line_y2=?"]
        values += [line.x1, line.y1, line.x2, line.y2]

    if cfg.polygon is not None:
        if len(cfg.polygon) < 3:
            raise HTTPException(status_code=422, detail="Area (polygon) minimal butuh 3 titik.")
        for pt in cfg.polygon:
            if len(pt) != 2 or not (0.0 <= pt[0] <= 1.0) or not (0.0 <= pt[1] <= 1.0):
                raise HTTPException(status_code=422, detail="Setiap titik polygon harus [x,y] dengan nilai 0-1.")
        if worker:
            worker.set_count_polygon(cfg.polygon)
        set_clauses.append("count_polygon=?"); values.append(json.dumps(cfg.polygon))

    if cfg.line_style is not None or cfg.line_thickness is not None:
        if cfg.line_style is not None and cfg.line_style not in ("rope", "solid", "dashed"):
            raise HTTPException(status_code=422, detail="line_style harus 'rope', 'dashed', atau 'solid'.")
        if worker:
            worker.set_count_line_style(cfg.line_style, cfg.line_thickness)
        if cfg.line_style is not None:
            set_clauses.append("count_line_style=?"); values.append(cfg.line_style)
        if cfg.line_thickness is not None:
            thickness = max(1.0, min(6.0, float(cfg.line_thickness)))
            set_clauses.append("count_line_thickness=?"); values.append(thickness)

    if set_clauses:
        values.append(cam_id)
        conn.execute(f"UPDATE cameras SET {', '.join(set_clauses)} WHERE id=?", values)
        conn.commit()
    conn.close()

    if worker:
        return {"status": "ok", "camera_active": True, **worker.get_counting_config()}
    return {"status": "ok", "camera_active": False,
            **_counting_config_from_row(_camera_row_or_404(cam_id))}


@app.get("/api/cameras/{cam_id}/counting/stats")
async def get_counting_stats(cam_id: int):
    """
    Statistik counting berjalan (per kategori: person/car/motorcycle/bus/truck/
    bicycle/animal), masing-masing dengan jumlah 'in' (masuk) dan 'out' (keluar)
    berdasarkan arah lintasan garis virtual.
    """
    worker = camera_manager.get(cam_id)
    if worker:
        with worker.lock:
            counts = dict(worker.counts)
    else:
        # Kamera nonaktif: angka terakhir tetap tersimpan di tabel camera_counts,
        # jadi tampilkan itu daripada melempar 404 dan bikin data terlihat hilang.
        _camera_row_or_404(cam_id)
        conn = get_conn()
        row = conn.execute("SELECT counts_json FROM camera_counts WHERE camera_id=?", (cam_id,)).fetchone()
        conn.close()
        try:
            counts = json.loads(row["counts_json"]) if row and row["counts_json"] else {}
        except Exception:
            counts = {}
    total_in = sum(c.get("in", 0) for c in counts.values())
    total_out = sum(c.get("out", 0) for c in counts.values())
    return {
        "camera_id": cam_id,
        "counts": counts,
        "labels": COUNTING_LABELS_ID,
        "total_in": total_in,
        "total_out": total_out,
    }


@app.post("/api/cameras/{cam_id}/counting/reset")
async def reset_counting_stats(cam_id: int):
    """Reset semua angka counting kamera ini kembali ke 0 (mis. mulai hari baru)."""
    _camera_row_or_404(cam_id)
    worker = camera_manager.get(cam_id)
    if not worker:
        return {"status": "ok", "camera_active": False,
                "detail": "Kamera sedang nonaktif — tidak ada hitungan berjalan."}
    worker.reset_counts()
    return {"status": "ok"}


# =========================================================
# 🅿️ PARKING DETECTION (poligon custom per slot)
# =========================================================
# Berbeda dari counting (1 garis/area per kamera), parking mendukung BANYAK slot
# sekaligus per kamera, masing-masing dengan bentuk poligon custom sendiri (garis
# lot yang bisa digambar bebas, tidak harus persegi). Status terisi/kosong tiap
# slot dihitung dari posisi kendaraan hasil YOLO yang jatuh di dalam poligonnya.

def _camera_row_or_404(cam_id: int) -> dict:
    """Ambil baris kamera dari DB. Dipakai endpoint konfigurasi supaya kamera yang
    sedang NONAKTIF (belum punya worker) tetap bisa diatur — worker hanya dibuat
    untuk kamera enabled=1, jadi bergantung pada worker bikin setting tidak bisa
    dibuka/disimpan selama kamera dimatikan."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan")
    return dict(row)


def _counting_config_from_row(row: dict) -> dict:
    """Bentuk dict yang sama persis dengan CameraWorker.get_counting_config(),
    tapi dibaca dari DB (untuk kamera yang workernya belum jalan)."""
    try:
        polygon = json.loads(row.get("count_polygon") or "[]")
    except Exception:
        polygon = []
    try:
        thickness = float(row.get("count_line_thickness") or 2.0)
    except Exception:
        thickness = 2.0
    return {
        "shape": row.get("count_shape") or "line",
        "direction": row.get("count_direction") or "both",
        "line": {
            "x1": float(row.get("count_line_x1") if row.get("count_line_x1") is not None else 0.1),
            "y1": float(row.get("count_line_y1") if row.get("count_line_y1") is not None else 0.5),
            "x2": float(row.get("count_line_x2") if row.get("count_line_x2") is not None else 0.9),
            "y2": float(row.get("count_line_y2") if row.get("count_line_y2") is not None else 0.5),
        },
        "polygon": polygon,
        "line_style": row.get("count_line_style") or "rope",
        "line_thickness": thickness,
    }


def _parking_slots_from_db(cam_id: int) -> list:
    """Daftar slot parkir langsung dari DB, tanpa status terisi/kosong (butuh worker)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM parking_slots WHERE camera_id=? ORDER BY id", (cam_id,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        try:
            poly = json.loads(r["polygon"] or "[]")
        except Exception:
            poly = []
        out.append({"id": r["id"], "label": r["label"], "polygon": poly,
                    "occupied": None, "changed_at": None})
    return out


def _validate_parking_polygon(polygon: list):
    if not polygon or len(polygon) < 3:
        raise HTTPException(status_code=422, detail="Poligon slot parkir minimal butuh 3 titik.")
    for pt in polygon:
        if len(pt) != 2 or not (0.0 <= pt[0] <= 1.0) or not (0.0 <= pt[1] <= 1.0):
            raise HTTPException(status_code=422, detail="Setiap titik poligon harus [x,y] dengan nilai 0-1.")


@app.get("/api/cameras/{cam_id}/parking/slots")
async def list_parking_slots(cam_id: int):
    """Daftar semua slot parkir kamera ini + status terisi/kosong terkini.
    Kalau kamera sedang nonaktif, slot tetap dikembalikan dari DB (occupied=None)
    supaya admin bisa menggambar/mengubah area parkir kapan saja."""
    worker = camera_manager.get(cam_id)
    if worker:
        return {"camera_id": cam_id, "camera_active": True, "slots": worker.get_parking_status()}
    _camera_row_or_404(cam_id)
    return {"camera_id": cam_id, "camera_active": False, "slots": _parking_slots_from_db(cam_id)}


@app.get("/api/cameras/{cam_id}/parking/status")
async def get_parking_status(cam_id: int):
    """Endpoint ringan buat dipoll dashboard tiap beberapa detik (overlay live +
    badge ringkasan) — isinya sama seperti /parking/slots, nama lebih jelas."""
    return await list_parking_slots(cam_id)


@app.post("/api/cameras/{cam_id}/parking/slots")
async def create_parking_slot(cam_id: int, item: ParkingSlotCreate):
    """Tambah 1 slot parkir baru dengan poligon custom hasil gambar admin di dashboard."""
    _camera_row_or_404(cam_id)
    worker = camera_manager.get(cam_id)
    _validate_parking_polygon(item.polygon)

    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO parking_slots (camera_id, label, polygon) VALUES (?, ?, ?)",
        (cam_id, item.label.strip(), json.dumps(item.polygon)),
    )
    new_id = cur.lastrowid
    if not item.label.strip():
        conn.execute("UPDATE parking_slots SET label=? WHERE id=?", (f"Slot {new_id}", new_id))
    conn.commit()
    conn.close()

    if worker:
        worker.reload_parking_slots()
        return {"status": "ok", "slots": worker.get_parking_status()}
    return {"status": "ok", "camera_active": False, "slots": _parking_slots_from_db(cam_id)}


@app.put("/api/cameras/{cam_id}/parking/slots/{slot_id}")
async def update_parking_slot(cam_id: int, slot_id: int, item: ParkingSlotUpdate):
    """Ubah label dan/atau bentuk poligon slot parkir yang sudah ada (mis. drag ulang
    titik-titiknya lewat dashboard lalu simpan)."""
    _camera_row_or_404(cam_id)
    worker = camera_manager.get(cam_id)

    conn = get_conn()
    row = conn.execute("SELECT * FROM parking_slots WHERE id=? AND camera_id=?", (slot_id, cam_id)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Slot parkir tidak ditemukan")

    set_clauses, values = [], []
    if item.label is not None:
        set_clauses.append("label=?"); values.append(item.label.strip() or row["label"])
    if item.polygon is not None:
        _validate_parking_polygon(item.polygon)
        set_clauses.append("polygon=?"); values.append(json.dumps(item.polygon))
    if set_clauses:
        values.append(slot_id)
        conn.execute(f"UPDATE parking_slots SET {', '.join(set_clauses)} WHERE id=?", values)
        conn.commit()
    conn.close()

    if worker:
        worker.reload_parking_slots()
        return {"status": "ok", "slots": worker.get_parking_status()}
    return {"status": "ok", "camera_active": False, "slots": _parking_slots_from_db(cam_id)}


@app.delete("/api/cameras/{cam_id}/parking/slots/{slot_id}")
async def delete_parking_slot(cam_id: int, slot_id: int):
    _camera_row_or_404(cam_id)
    worker = camera_manager.get(cam_id)
    conn = get_conn()
    conn.execute("DELETE FROM parking_slots WHERE id=? AND camera_id=?", (slot_id, cam_id))
    conn.commit()
    conn.close()
    if worker:
        worker.reload_parking_slots()
    return {"status": "ok"}


# =========================================================
# 🎬 REKAMAN (NVR) — rekam langsung dari RTSP via ffmpeg
# =========================================================

def _get_camera_row_or_404(cam_id: int) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan")
    return dict(row)


@app.get("/api/cameras/{cam_id}/recording/settings")
async def get_recording_settings(cam_id: int):
    """Setting rekaman kamera ini + status live (sedang merekam / tidak / error)."""
    cam = _get_camera_row_or_404(cam_id)
    status = recording_manager.get_status(cam_id)
    return {
        "camera_id": cam_id,
        "record_enabled": bool(cam["record_enabled"]),
        "resolution": cam["record_resolution"] or "original",
        "crf": cam["record_crf"] or RECORD_CRF_DEFAULT,
        "segment_min": cam["record_segment_min"] or RECORD_SEGMENT_MIN_DEFAULT,
        "retention_days": cam["record_retention_days"] or RECORD_RETENTION_DAYS_DEFAULT,
        "resolution_options": list(RECORD_RESOLUTIONS.keys()),
        "ffmpeg_available": bool(FFMPEG_BIN),
        **status,
    }


@app.post("/api/cameras/{cam_id}/recording/settings")
async def update_recording_settings(cam_id: int, cfg: RecordingSettingsUpdate):
    """
    Simpan setting rekaman (resolusi/kompresi/durasi segmen/retensi). Kalau kamera
    SEDANG merekam, rekaman otomatis di-restart pakai setting baru supaya langsung
    berlaku tanpa admin harus stop-lalu-start manual.
    """
    cam = _get_camera_row_or_404(cam_id)

    if cfg.resolution is not None and cfg.resolution not in RECORD_RESOLUTIONS:
        raise HTTPException(status_code=422, detail=f"resolution harus salah satu dari: {list(RECORD_RESOLUTIONS.keys())}")
    if cfg.crf is not None and not (0 <= cfg.crf <= 51):
        raise HTTPException(status_code=422, detail="crf harus 0-51")
    if cfg.segment_min is not None and not (1 <= cfg.segment_min <= 180):
        raise HTTPException(status_code=422, detail="segment_min harus 1-180 menit")
    if cfg.retention_days is not None and not (0 <= cfg.retention_days <= 365):
        raise HTTPException(status_code=422, detail="retention_days harus 0-365 (0 = simpan selamanya)")

    set_clauses, values = [], []
    if cfg.resolution is not None:
        set_clauses.append("record_resolution=?"); values.append(cfg.resolution)
    if cfg.crf is not None:
        set_clauses.append("record_crf=?"); values.append(cfg.crf)
    if cfg.segment_min is not None:
        set_clauses.append("record_segment_min=?"); values.append(cfg.segment_min)
    if cfg.retention_days is not None:
        set_clauses.append("record_retention_days=?"); values.append(cfg.retention_days)

    if set_clauses:
        conn = get_conn()
        values.append(cam_id)
        conn.execute(f"UPDATE cameras SET {', '.join(set_clauses)} WHERE id=?", values)
        conn.commit()
        conn.close()

    # Kalau lagi merekam, restart dgn setting baru supaya langsung efektif
    status = recording_manager.get_status(cam_id)
    if status.get("recording"):
        cam = _get_camera_row_or_404(cam_id)
        recording_manager.stop(cam_id)
        time.sleep(0.3)
        recording_manager.start(
            cam_id, cam["name"], cam["rtsp_url"],
            resolution=cam["record_resolution"] or "original",
            crf=cam["record_crf"] or RECORD_CRF_DEFAULT,
            segment_min=cam["record_segment_min"] or RECORD_SEGMENT_MIN_DEFAULT,
        )

    return {"status": "ok", **(await get_recording_settings(cam_id))}


@app.post("/api/cameras/{cam_id}/recording/start")
async def start_recording(cam_id: int):
    """Mulai rekam kamera ini (langsung dari RTSP, terpisah dari live-AI)."""
    cam = _get_camera_row_or_404(cam_id)
    if not FFMPEG_BIN:
        raise HTTPException(status_code=500, detail="ffmpeg tidak tersedia di server ini — install ffmpeg dulu.")
    try:
        recording_manager.start(
            cam_id, cam["name"], cam["rtsp_url"],
            resolution=cam["record_resolution"] or "original",
            crf=cam["record_crf"] or RECORD_CRF_DEFAULT,
            segment_min=cam["record_segment_min"] or RECORD_SEGMENT_MIN_DEFAULT,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    conn = get_conn()
    conn.execute("UPDATE cameras SET record_enabled=1 WHERE id=?", (cam_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "recording": True}


@app.post("/api/cameras/{cam_id}/recording/stop")
async def stop_recording(cam_id: int):
    """Hentikan rekaman kamera ini."""
    _get_camera_row_or_404(cam_id)
    recording_manager.stop(cam_id)
    conn = get_conn()
    conn.execute("UPDATE cameras SET record_enabled=0 WHERE id=?", (cam_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "recording": False}


@app.get("/api/cameras/{cam_id}/recording/status")
async def recording_status(cam_id: int):
    _get_camera_row_or_404(cam_id)
    return {"camera_id": cam_id, **recording_manager.get_status(cam_id)}


@app.get("/api/cameras/{cam_id}/recordings")
async def list_recordings(cam_id: int):
    """
    Daftar file rekaman kamera ini dari SEMUA lokasi storage yang pernah/sedang
    dikonfigurasi (bukan cuma yang aktif sekarang) — supaya rekaman sebelum
    terjadinya failover tetap kelihatan & bisa diputar/diunduh, tidak "hilang".
    """
    _get_camera_row_or_404(cam_id)
    files = []
    total_size = 0
    for root in storage_manager.get_all_roots():
        cam_dir = os.path.join(root, str(cam_id))
        if not os.path.isdir(cam_dir):
            continue
        for fname in os.listdir(cam_dir):
            if not RECORD_FILENAME_RE.match(fname):
                continue  # abaikan file asing/tidak dikenal di folder ini
            fpath = os.path.join(cam_dir, fname)
            try:
                stat = os.stat(fpath)
            except OSError:
                continue
            total_size += stat.st_size
            # nama file formatnya YYYYMMDD_HHMMSS.mp4 (dibuat ffmpeg -strftime) -> parse jadi waktu mulai
            try:
                started = datetime.strptime(fname.replace(".mp4", ""), "%Y%m%d_%H%M%S")
            except ValueError:
                started = None
            files.append({
                "filename": fname,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "started_at": started.isoformat() if started else None,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "storage_root": root,
            })
    files.sort(key=lambda f: f["filename"], reverse=True)
    return {"camera_id": cam_id, "files": files, "total_size_mb": round(total_size / (1024 * 1024), 2)}


def _safe_recording_path(cam_id: int, filename: str) -> str:
    """Validasi ketat nama file rekaman supaya endpoint download/hapus tidak bisa
    dipakai buat path traversal (mis. '../../etc/passwd'). Dicari di SEMUA lokasi
    storage yang dikonfigurasi (file lama bisa ada di disk yang sudah di-failover-kan)."""
    if not RECORD_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Nama file tidak valid.")
    for root in storage_manager.get_all_roots():
        fpath = os.path.join(root, str(cam_id), filename)
        if os.path.isfile(fpath):
            return fpath
    raise HTTPException(status_code=404, detail="File rekaman tidak ditemukan.")


@app.get("/api/cameras/{cam_id}/recordings/{filename}")
async def download_recording(cam_id: int, filename: str):
    """Download/putar 1 file rekaman (dipakai juga sebagai src <video> player di dashboard)."""
    _get_camera_row_or_404(cam_id)
    fpath = _safe_recording_path(cam_id, filename)
    return FileResponse(fpath, media_type="video/mp4", filename=filename)


@app.delete("/api/cameras/{cam_id}/recordings/{filename}")
async def delete_recording(cam_id: int, filename: str):
    _get_camera_row_or_404(cam_id)
    fpath = _safe_recording_path(cam_id, filename)
    try:
        os.remove(fpath)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus file: {e}")
    return {"status": "ok"}


@app.get("/api/recordings/disk_usage")
async def recordings_disk_usage():
    """Ringkasan pemakaian disk rekaman per kamera + per lokasi storage, buat admin
    pantau kapasitas tanpa perlu buka terminal server."""
    conn = get_conn()
    cams = {row["id"]: row["name"] for row in conn.execute("SELECT id, name FROM cameras").fetchall()}
    conn.close()

    size_by_cam: dict = {}
    grand_total = 0
    for root in storage_manager.get_all_roots():
        if not os.path.isdir(root):
            continue
        for cam_dir_name in os.listdir(root):
            cam_dir = os.path.join(root, cam_dir_name)
            if not os.path.isdir(cam_dir) or not cam_dir_name.isdigit():
                continue
            size = sum(
                os.path.getsize(os.path.join(cam_dir, f))
                for f in os.listdir(cam_dir) if RECORD_FILENAME_RE.match(f)
            )
            grand_total += size
            size_by_cam[int(cam_dir_name)] = size_by_cam.get(int(cam_dir_name), 0) + size

    per_camera = [
        {"camera_id": cid, "camera_name": cams.get(cid, "(sudah dihapus)"), "size_mb": round(sz / (1024 * 1024), 2)}
        for cid, sz in size_by_cam.items()
    ]

    storage_status = storage_manager.status()
    active = next((p for p in storage_status["paths"] if p["is_active"]), None)

    return {
        "per_camera": sorted(per_camera, key=lambda c: -c["size_mb"]),
        "grand_total_mb": round(grand_total / (1024 * 1024), 2),
        "disk_total_gb": active.get("total_gb") if active else None,
        "disk_free_gb": active.get("free_gb") if active else None,
        "storage_paths": storage_status["paths"],
    }


# =========================================================
# 💾 STORAGE MANAGEMENT — pilih lokasi penyimpanan + failover
# =========================================================

@app.get("/api/storage/disks")
async def list_available_disks():
    """
    Deteksi drive/partisi fisik yang terpasang di server ini (lintas platform:
    C:\\, D:\\ dst di Windows; /, /mnt/xxx, /media/xxx di Linux/macOS) lengkap sisa
    ruangnya — dipakai dashboard supaya admin tinggal PILIH dari daftar, bukan
    ngetik path manual (mengurangi salah ketik / salah drive).
    """
    if not PSUTIL_AVAILABLE:
        return {
            "disks": [],
            "note": "psutil tidak terpasang di server — deteksi drive otomatis tidak tersedia. "
                    "Admin tetap bisa isi path folder manual di kolom 'Tambah lokasi custom'.",
        }
    disks = []
    seen = set()
    for part in psutil.disk_partitions(all=False):
        if part.mountpoint in seen:
            continue
        seen.add(part.mountpoint)
        try:
            usage = shutil.disk_usage(part.mountpoint)
        except Exception:
            continue  # drive tidak siap diakses (mis. CD-ROM kosong) -> lewati
        disks.append({
            "mountpoint": part.mountpoint,
            "device": part.device,
            "fstype": part.fstype,
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "used_pct": round(usage.used / usage.total * 100, 1) if usage.total else 0,
        })
    disks.sort(key=lambda d: -d["free_gb"])
    return {"disks": disks}


@app.get("/api/storage/paths")
async def get_storage_paths():
    """Daftar lokasi penyimpanan yang dikonfigurasi (urut prioritas) + kesehatan
    tiap lokasi + lokasi mana yang aktif dipakai sekarang + riwayat failover."""
    return storage_manager.status()


@app.post("/api/storage/paths/test")
async def test_storage_path(item: StoragePathTest):
    """Cek 1 path SEBELUM disimpan — apakah bisa ditulis & berapa sisa ruangnya."""
    result = StorageManager.check_health(item.path)
    return {"path": item.path, **result}


@app.post("/api/storage/paths")
async def update_storage_paths(update: StoragePathsUpdate):
    """
    Simpan daftar lokasi penyimpanan LENGKAP beserta urutan prioritasnya.
    Index ke-0 = primary (dipakai duluan selama sehat), index berikutnya = cadangan
    failover berurutan. Minimal 1 lokasi wajib diisi.
    """
    if not update.paths:
        raise HTTPException(status_code=422, detail="Minimal harus ada 1 lokasi penyimpanan.")

    cleaned = []
    for item in update.paths:
        p = os.path.abspath(item.path.strip())
        if not p or p in [c["path"] for c in cleaned]:
            continue
        cleaned.append({"path": p, "label": (item.label or "").strip()[:80]})
    if not cleaned:
        raise HTTPException(status_code=422, detail="Tidak ada path valid setelah divalidasi.")

    conn = get_conn()
    conn.execute("DELETE FROM storage_paths")
    for idx, item in enumerate(cleaned):
        conn.execute(
            "INSERT INTO storage_paths (path, priority, label, enabled) VALUES (?, ?, ?, 1)",
            (item["path"], idx, item["label"]),
        )
    conn.commit()
    conn.close()

    storage_manager._load_paths()
    changed = storage_manager.recheck()
    if changed:
        recording_manager.relocate_all(storage_manager.get_recordings_root())

    return storage_manager.status()


# =========================================================
# 🎬 VIDEO FEED PER-KAMERA
# =========================================================

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int):
    worker = camera_manager.get(cam_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")

    async def gen():
        # --- 🎥 Streaming event-driven, bukan poll buta ---
        # Dulu: cek tiap 50ms TANPA peduli apakah frame benar-benar baru → kadang kirim
        # frame lama berkali-kali (buang bandwidth, tidak nambah kemulusan), kadang telat
        # ~50ms kirim frame yang sudah siap (nambah rasa 'lag'/patah).
        # Sekarang: cek tiap 10ms apakah frame_version berubah. Begitu berubah, LANGSUNG
        # kirim (delay maksimum cuma ~10ms, jauh di bawah ambang yang terasa mata manusia).
        # Kalau belum berubah, tidak kirim apa² (hemat CPU & bandwidth, tidak ada duplikat).
        last_sent_version = -1
        while True:
            frame_bytes, version = worker.get_jpeg_with_version()
            if frame_bytes is not None and version != last_sent_version:
                last_sent_version = version
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cameras/{cam_id}/alerts")
async def get_camera_alerts(cam_id: int):
    worker = camera_manager.get(cam_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")
    with worker.lock:
        return {"alerts": worker.latest_alerts}


@app.get("/api/cameras/{cam_id}/detections")
async def get_live_detections(cam_id: int):
    """
    Box deteksi AI TERBARU (posisi dinormalisasi 0-1), dipakai dashboard untuk
    menggambar overlay kotak 3D dinamis di atas video — dipoll ringan tiap
    beberapa ratus ms, jauh lebih murah daripada streaming metadata lewat websocket.
    """
    worker = camera_manager.get(cam_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")
    with worker.lock:
        boxes = list(worker.latest_detections)
    return {"camera_id": cam_id, "ts": time.time(), "boxes": boxes}


@app.get("/api/alerts/all")
async def get_all_alerts():
    """Gabungan alert dari semua kamera, untuk feed global."""
    combined = []
    for cam_id, worker in camera_manager.all().items():
        with worker.lock:
            for a in worker.latest_alerts:
                combined.append({**a, "camera_id": cam_id, "camera_name": worker.name})
    combined.sort(key=lambda a: a["time"], reverse=True)
    return {"alerts": combined[:100]}


# =========================================================
# 📸 GALERI CAPTURE (playback hasil capture visual — persisten di disk)
# =========================================================
# Berbeda dari /api/alerts (in-memory, maks 50, hilang saat restart), endpoint di
# bawah ini membaca snapshot yang sudah ditulis permanen ke CAPTURES_DIR_DEFAULT
# oleh CameraWorker._make_alert(), lengkap dengan sidecar .json metadata-nya.

def _read_capture_meta(cam_dir: str, fname: str) -> dict:
    meta_path = os.path.join(cam_dir, fname[:-4] + ".json")
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/api/cameras/{cam_id}/captures")
async def list_captures(cam_id: int, type: Optional[str] = None, limit: int = 200):
    """Daftar snapshot capture kamera ini (terbaru dulu), opsional difilter per jenis
    (face/smoking/fire/behavior_fall/counting) — dipakai galeri 'playback capture' di dashboard."""
    _get_camera_row_or_404(cam_id)
    cam_dir = os.path.join(CAPTURES_DIR_DEFAULT, str(cam_id))
    files = []
    if os.path.isdir(cam_dir):
        for fname in os.listdir(cam_dir):
            if not CAPTURE_FILENAME_RE.match(fname):
                continue
            fpath = os.path.join(cam_dir, fname)
            try:
                stat = os.stat(fpath)
            except OSError:
                continue
            meta = _read_capture_meta(cam_dir, fname)
            entry = {
                "filename": fname,
                "type": meta.get("type", fname.split("_", 2)[-1].rsplit("_", 1)[0]),
                "name": meta.get("name", ""),
                "created_at": meta.get("created_at") or datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
            if type and entry["type"] != type:
                continue
            files.append(entry)
    files.sort(key=lambda f: f["filename"], reverse=True)
    return {"camera_id": cam_id, "captures": files[:max(1, min(limit, 1000))], "total": len(files)}


def _safe_capture_path(cam_id: int, filename: str) -> str:
    if not CAPTURE_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Nama file tidak valid.")
    fpath = os.path.join(CAPTURES_DIR_DEFAULT, str(cam_id), filename)
    if os.path.isfile(fpath):
        return fpath
    raise HTTPException(status_code=404, detail="File capture tidak ditemukan.")


@app.get("/api/cameras/{cam_id}/captures/{filename}")
async def get_capture_image(cam_id: int, filename: str):
    """Ambil gambar 1 capture (dipakai sebagai src <img> di galeri/lightbox dashboard)."""
    _get_camera_row_or_404(cam_id)
    fpath = _safe_capture_path(cam_id, filename)
    return FileResponse(fpath, media_type="image/jpeg", filename=filename)


@app.delete("/api/cameras/{cam_id}/captures/{filename}")
async def delete_capture(cam_id: int, filename: str):
    _get_camera_row_or_404(cam_id)
    fpath = _safe_capture_path(cam_id, filename)
    try:
        os.remove(fpath)
        meta_path = fpath[:-4] + ".json"
        if os.path.isfile(meta_path):
            os.remove(meta_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus file: {e}")
    return {"status": "ok"}


@app.get("/api/captures/all")
async def list_all_captures(type: Optional[str] = None, limit: int = 100):
    """Galeri gabungan semua kamera — dipakai tab 'Capture' global di dashboard."""
    conn = get_conn()
    cams = {row["id"]: row["name"] for row in conn.execute("SELECT id, name FROM cameras").fetchall()}
    conn.close()

    combined = []
    if os.path.isdir(CAPTURES_DIR_DEFAULT):
        for cam_dir_name in os.listdir(CAPTURES_DIR_DEFAULT):
            if not cam_dir_name.isdigit():
                continue
            cam_id = int(cam_dir_name)
            cam_dir = os.path.join(CAPTURES_DIR_DEFAULT, cam_dir_name)
            if not os.path.isdir(cam_dir):
                continue
            for fname in os.listdir(cam_dir):
                if not CAPTURE_FILENAME_RE.match(fname):
                    continue
                meta = _read_capture_meta(cam_dir, fname)
                entry_type = meta.get("type", fname.split("_", 2)[-1].rsplit("_", 1)[0])
                if type and entry_type != type:
                    continue
                fpath = os.path.join(cam_dir, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                except OSError:
                    continue
                combined.append({
                    "camera_id": cam_id,
                    "camera_name": cams.get(cam_id, meta.get("camera_name", "(kamera dihapus)")),
                    "filename": fname,
                    "type": entry_type,
                    "name": meta.get("name", ""),
                    "created_at": meta.get("created_at") or datetime.fromtimestamp(mtime).isoformat(),
                })
    combined.sort(key=lambda c: c["created_at"], reverse=True)
    return {"captures": combined[:max(1, min(limit, 500))], "total": len(combined)}


# =========================================================
# 👤 FACE REGISTRATION (per-kamera capture + upload foto)
# =========================================================

def analyze_occlusion_public(canon_gray):
    from enpidix_face_engine import analyze_occlusion
    o = analyze_occlusion(canon_gray)
    o["arti"] = {"none": "wajah terbuka", "mask": "bermasker",
                 "headwear": "bertopi/berkerudung",
                 "mask_headwear": "bermasker + bertopi/berkerudung"}.get(o["state"], "?")
    return o


def _extract_largest_face_gray(image_bytes: bytes) -> np.ndarray:
    """
    Decode bytes gambar hasil upload, deteksi wajah pakai Haar cascade yang sama
    dengan yang dipakai live detection, lalu kembalikan crop grayscale wajah
    TERBESAR (paling dominan) di foto tsb — konsisten dengan format yang dipakai
    face_crop dari alur capture kamera, supaya bisa langsung masuk training LBPH.
    Melempar ValueError kalau file bukan gambar valid atau tidak ada wajah terdeteksi.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File bukan gambar yang valid.")

    dets = face_engine.detect(img)
    if not dets:
        raise ValueError("Tidak ada wajah terdeteksi. Gunakan foto close-up wajah yang jelas.")

    # Kembalikan crop yang SUDAH di-align supaya seluruh training set berada di
    # sistem koordinat yang sama, apapun sumber fotonya (HP, KTP, atau kamera).
    return face_engine.align(img, dets[0])


@app.post("/api/faces/register/upload")
async def register_face_upload(name: str = Form(...), files: list[UploadFile] = File(...)):
    """
    Registrasi wajah baru ke database lewat UPLOAD FOTO (tanpa perlu live capture
    dari kamera). Mendukung upload beberapa foto sekaligus untuk nama yang sama —
    makin banyak sampel foto (sudut/pencahayaan berbeda), makin akurat pengenalan
    LBPH-nya. Setiap foto yang berhasil diproses langsung ditambahkan ke tabel
    `persons` dan model recognizer di-retrain ulang.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nama tidak boleh kosong.")
    if not files:
        raise HTTPException(status_code=400, detail="Minimal 1 foto wajib diupload.")

    saved, failed = [], []
    conn = get_conn()
    try:
        for f in files:
            content = await f.read()
            try:
                face_gray = _extract_largest_face_gray(content)
            except ValueError as e:
                failed.append({"filename": f.filename, "reason": str(e)})
                continue

            conn.execute("INSERT INTO persons (name, image_path) VALUES (?, ?)", (name, ""))
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            img_path = os.path.join(FACES_DIR, f"{new_id}_{name.replace(' ', '_')}.jpg")
            cv2.imwrite(img_path, face_gray)
            conn.execute("UPDATE persons SET image_path=? WHERE id=?", (img_path, new_id))
            saved.append({"id": new_id, "filename": f.filename})
        conn.commit()
    finally:
        conn.close()

    count = train_recognizer()

    if not saved:
        raise HTTPException(
            status_code=422,
            detail={"message": "Tidak ada foto yang berhasil diproses (wajah tidak terdeteksi).", "failed": failed},
        )

    return {
        "status": "ok",
        "name": name,
        "saved": saved,
        "failed": failed,
        "total_trained_faces": count,
    }


@app.post("/api/cameras/{cam_id}/register/capture")
async def request_capture(cam_id: int):
    worker = camera_manager.get(cam_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")

    worker.request_capture()
    waited = 0.0
    while worker.pending_capture["result"] is None and waited < 5.0:
        await asyncio.sleep(0.1)
        waited += 0.1

    result = worker.pending_capture["result"]
    if result is None:
        raise HTTPException(status_code=504, detail="Timeout menunggu kamera.")
    if not result["ok"]:
        raise HTTPException(status_code=422, detail=result["reason"])

    _, buf = cv2.imencode(".jpg", result["frame_crop"])
    preview_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    token = str(int(time.time() * 1000))
    worker.capture_cache[token] = result["face_crop"]
    return {"status": "ok", "preview": preview_b64, "token": token}


@app.post("/api/cameras/{cam_id}/register/save")
async def save_face(cam_id: int, name: str = Form(...), token: str = Form(...)):
    worker = camera_manager.get(cam_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")

    face_gray = worker.capture_cache.pop(token, None)
    if face_gray is None:
        raise HTTPException(status_code=400, detail="Token tidak valid. Ambil foto ulang.")
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nama tidak boleh kosong.")

    conn = get_conn()
    conn.execute("INSERT INTO persons (name, image_path) VALUES (?, ?)", (name, ""))
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    img_path = os.path.join(FACES_DIR, f"{new_id}_{name.replace(' ', '_')}.jpg")
    cv2.imwrite(img_path, face_gray)
    conn.execute("UPDATE persons SET image_path=? WHERE id=?", (img_path, new_id))
    conn.commit()
    conn.close()

    count = train_recognizer()
    return {"status": "ok", "id": new_id, "name": name, "total_trained_faces": count}


@app.get("/api/smoking/status")
async def smoking_status():
    """Kondisi deteksi merokok: model pose aktif atau tidak, ambang, dan siapa
    yang sedang dipantau di tiap kamera beserta jumlah isapan yang tercatat."""
    per_cam = []
    for w in camera_manager.all().values():
        try:
            per_cam.append({
                "camera_id": w.id, "camera_name": w.name,
                "aktif": bool(w.ai_settings.get("smoking_detection")),
                "dipantau": w.smoking_engine.snapshot(),
            })
        except Exception:
            pass
    pose_ok = os.path.exists(smoking_cfg.POSE_MODEL_PATH)
    return {
        "model_pose": pose_ok,
        "model_pose_path": smoking_cfg.POSE_MODEL_PATH,
        "peringatan": ([] if pose_ok else [
            "Model pose tidak ditemukan. Bukti GESTUR memakai perkiraan gerakan, "
            "yang jauh lebih lemah — perokok asli banyak yang lolos. Tanpa model ini, "
            "manfaat utamanya hanya menghapus false positive HP/gelas."]),
        "ambang": {
            "skor_minimal": smoking_cfg.SCORE_THRESHOLD,
            "bukti_kuat_minimal": smoking_cfg.STRONG_CUE_MIN,
            "isapan_minimal": smoking_cfg.MIN_CYCLES,
            "durasi_isapan_detik": [smoking_cfg.DRAG_MIN_SEC, smoking_cfg.DRAG_MAX_SEC],
            "jendela_detik": smoking_cfg.WINDOW_SEC,
        },
        "penyangkal": sorted(SMOKING_CONFUSER_CLASSES),
        "kamera": per_cam,
    }


@app.patch("/api/smoking/settings")
async def smoking_settings(payload: dict = Body(...)):
    """Setel ambang deteksi merokok tanpa restart.

    Kunci yang diterima: skor_minimal, bukti_kuat_minimal, isapan_minimal,
    durasi_isapan_min, durasi_isapan_max, jendela_detik, hukuman_penyangkal.

    Kalau perokok asli lolos: turunkan isapan_minimal ke 1 atau lebarkan
    durasi_isapan_max. Kalau orang menelepon masih ke-flag: naikkan
    hukuman_penyangkal atau turunkan durasi_isapan_max.
    """
    m = {
        "skor_minimal": ("SCORE_THRESHOLD", 0.0, 1.0),
        "bukti_kuat_minimal": ("STRONG_CUE_MIN", 0.0, 1.0),
        "isapan_minimal": ("MIN_CYCLES", 1, 10),
        "durasi_isapan_min": ("DRAG_MIN_SEC", 0.1, 5.0),
        "durasi_isapan_max": ("DRAG_MAX_SEC", 0.5, 15.0),
        "jendela_detik": ("WINDOW_SEC", 10.0, 300.0),
        "hukuman_penyangkal": ("CONFUSER_PENALTY", 0.0, 1.0),
    }
    changed = {}
    for k, v in payload.items():
        if k not in m:
            raise HTTPException(status_code=422, detail=f"Kunci tidak dikenal: {k}")
        attr, lo, hi = m[k]
        if not (lo <= v <= hi):
            raise HTTPException(status_code=422, detail=f"{k} harus antara {lo}-{hi}")
        setattr(smoking_cfg, attr, type(getattr(smoking_cfg, attr))(v))
        changed[k] = v
    return {"status": "ok", "diubah": changed}


@app.get("/api/smoking/evidence/{camera_id}")
async def smoking_evidence(camera_id: int, limit: int = 20):
    """Daftar paket bukti merokok tersimpan, terbaru dulu.

    Tiap paket berisi frame beranotasi, crop orang, crop kepala diperbesar,
    cuplikan beberapa detik SEBELUM dan SESUDAH alert, plus bukti.json berisi
    rincian tiap isyarat. Pre-roll itu penting: saat skor akhirnya melewati
    ambang, isapan yang membentuk bukti sudah selesai beberapa detik sebelumnya.
    """
    base = os.path.join(CAPTURES_DIR_DEFAULT, "bukti_merokok", str(camera_id))
    if not os.path.isdir(base):
        return {"paket": []}
    out = []
    for name in sorted(os.listdir(base), reverse=True):
        d = os.path.join(base, name)
        meta_p = os.path.join(d, "bukti.json")
        if not os.path.isfile(meta_p):
            continue
        try:
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        meta["berkas"] = sorted(os.listdir(d))
        meta["pra_kejadian"] = len(os.listdir(os.path.join(d, "pra_kejadian"))) \
            if os.path.isdir(os.path.join(d, "pra_kejadian")) else 0
        meta["pasca_kejadian"] = len(os.listdir(os.path.join(d, "pasca_kejadian"))) \
            if os.path.isdir(os.path.join(d, "pasca_kejadian")) else 0
        out.append(meta)
        if len(out) >= limit:
            break
    return {"paket": out}


@app.get("/api/faces/engine/status")
async def face_engine_status():
    """Kondisi mesin wajah: model apa yang aktif, berapa sampel terlatih, ambang.

    Kalau "yunet": false, deteksi masih jatuh ke Haar dan wajah bermasker
    kemungkinan besar TIDAK AKAN TERDETEKSI — jalankan scripts/download_face_models.
    """
    st = face_engine.status()
    st["peringatan"] = []
    if not st["yunet"]:
        st["peringatan"].append(
            "Model YuNet tidak ditemukan. Deteksi memakai Haar cascade, yang gagal "
            "pada wajah bermasker. Jalankan scripts/download_face_models.bat (Windows) "
            "atau .sh (Linux/Jetson).")
    if not st["sface"]:
        st["peringatan"].append(
            "Model SFace tidak ditemukan. Pengenalan memakai ensemble LBPH — tetap "
            "berfungsi, tapi akurasinya lebih rendah dan makin lambat seiring "
            "bertambahnya orang terdaftar.")
    return st


@app.post("/api/faces/engine/retrain")
async def face_engine_retrain():
    """Latih ulang paksa. Dipakai setelah mengubah pengaturan augmentasi atau
    setelah menaruh file model ONNX yang baru diunduh."""
    count = train_recognizer()
    return {"status": "ok", "total_trained_faces": count, "stats": face_train_stats}


@app.post("/api/faces/engine/debug")
async def face_engine_debug(file: UploadFile = File(...)):
    """Bongkar sebuah foto: wajah terdeteksi di mana, oklusi terbaca apa, dan
    skor kecocokan per-region terhadap tiap orang terdaftar.

    Inilah alat untuk menyetel ambang di lokasi Anda sendiri. Kalau seseorang
    tidak dikenali, lihat 'skor' dan 'runner_up': kalau nama benar sudah di
    peringkat 1 tapi skornya di bawah FACE_ACCEPT_SIM, turunkan ambang itu.
    Kalau nama yang salah ikut tinggi, naikkan FACE_ACCEPT_MARGIN.
    """
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    dets = face_engine.detect(img)
    if not dets:
        return {"terdeteksi": 0, "catatan": "Tidak ada wajah terdeteksi.",
                "engine": face_engine.status()}

    out = []
    for d in dets:
        canon = face_engine.align(img, d)
        res = face_engine.recognize_frame(img, d, detail=True)
        res.pop("canon", None)
        out.append({
            "box": d["box"], "sumber_deteksi": d["source"],
            "skor_deteksi": round(d["score"], 3),
            "punya_landmark": d.get("landmarks") is not None,
            "oklusi": analyze_occlusion_public(canon),
            "hasil": res,
        })
    return {"terdeteksi": len(dets), "wajah": out,
            "ambang": {"accept_sim": face_engine.status()["accept_sim"],
                       "accept_margin": face_engine.status()["accept_margin"]}}


@app.get("/api/faces/engine/selftest")
async def face_engine_selftest():
    """Ambil setiap foto terdaftar, tempeli MASKER / TOPI / KERUDUNG sintetis,
    lalu ukur berapa persen yang masih dikenali dengan benar.

    Jalankan ini setelah mendaftarkan karyawan untuk melihat kondisi mana yang
    masih lemah SEBELUM mengandalkannya di lapangan. Perlu diingat: oklusi di
    sini sintetis, jadi angkanya optimis dibanding kondisi nyata — gunakan
    sebagai perbandingan relatif antar-kondisi, bukan janji akurasi absolut.
    """
    if not recognizer_ready:
        raise HTTPException(status_code=400, detail="Belum ada wajah terdaftar.")
    return {"hasil": face_engine.selftest(_person_rows()), "stats": face_train_stats}


@app.get("/api/faces")
async def list_faces():
    conn = get_conn()
    rows = conn.execute("SELECT id, name, image_path, created_at FROM persons ORDER BY id DESC").fetchall()
    conn.close()
    faces = []
    for r in rows:
        d = dict(r)
        # image_path tersimpan sebagai path lokal (mis. "registered_faces/3_Budi.jpg");
        # dikonversi ke URL yang bisa langsung dipakai <img src> lewat mount /faces.
        d["image_url"] = ("/faces/" + os.path.basename(d["image_path"])) if d["image_path"] else ""
        faces.append(d)
    return {"faces": faces}


@app.delete("/api/faces/{face_id}")
async def delete_face(face_id: int):
    conn = get_conn()
    row = conn.execute("SELECT image_path FROM persons WHERE id=?", (face_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Wajah tidak ditemukan")
    img_path = row["image_path"]
    conn.execute("DELETE FROM persons WHERE id=?", (face_id,))
    conn.commit()
    conn.close()
    if os.path.exists(img_path):
        os.remove(img_path)
    train_recognizer()
    return {"status": "ok"}


# =========================================================
# 📊 STATUS GLOBAL
# =========================================================

# =========================================================
# 🔄 LEGACY COMPAT ENDPOINTS (dari server.py single-camera lama)
# =========================================================
# Endpoint ini menunjuk ke kamera PERTAMA yang ada di sistem,
# supaya integrasi/script lama yang masih panggil /api/toggle atau
# /api/alerts (tanpa cam_id) tetap berfungsi tanpa perlu diubah.

def _get_first_camera_worker() -> Optional["CameraWorker"]:
    all_workers = camera_manager.all()
    if not all_workers:
        return None
    first_id = sorted(all_workers.keys())[0]
    return all_workers[first_id]


@app.post("/api/toggle")
async def legacy_toggle(feature: str, status: bool):
    """
    [LEGACY] Sama seperti /api/cameras/{id}/ai_toggle tapi otomatis
    pakai kamera pertama. Dipertahankan untuk kompatibilitas dengan
    server.py versi single-camera lama.
    """
    worker = _get_first_camera_worker()
    if not worker:
        raise HTTPException(status_code=404, detail="Belum ada kamera terdaftar di sistem.")
    if feature not in worker.ai_settings:
        raise HTTPException(status_code=400, detail=f"Fitur tidak dikenal: {feature}")

    worker.update_ai_settings(**{feature: status})

    col_map = {
        "face_recognition": "ai_face_recognition",
        "smoking_detection": "ai_smoking_detection",
        "fire_detection": "ai_fire_detection",
        "behavior_detection": "ai_behavior_detection",
    }
    conn = get_conn()
    conn.execute(f"UPDATE cameras SET {col_map[feature]}=? WHERE id=?", (1 if status else 0, worker.id))
    conn.commit()
    conn.close()

    return {"status": "ok", "camera_id": worker.id, "settings": worker.ai_settings}


@app.get("/api/alerts")
async def legacy_get_alerts():
    """[LEGACY] Alert dari kamera pertama saja. Untuk semua kamera, pakai /api/alerts/all."""
    worker = _get_first_camera_worker()
    if not worker:
        return {"alerts": []}
    with worker.lock:
        return {"alerts": worker.latest_alerts}


@app.get("/video_feed")
async def legacy_video_feed():
    """[LEGACY] Stream dari kamera pertama. Untuk multi-camera, pakai /video_feed/{id}."""
    worker = _get_first_camera_worker()
    if not worker:
        raise HTTPException(status_code=404, detail="Belum ada kamera terdaftar di sistem.")

    async def gen():
        last_sent_version = -1
        while True:
            frame_bytes, version = worker.get_jpeg_with_version()
            if frame_bytes is not None and version != last_sent_version:
                last_sent_version = version
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/register/capture")
async def legacy_request_capture():
    """[LEGACY] Capture wajah dari kamera pertama."""
    worker = _get_first_camera_worker()
    if not worker:
        raise HTTPException(status_code=404, detail="Belum ada kamera terdaftar di sistem.")
    return await request_capture(worker.id)


@app.post("/api/register/save")
async def legacy_save_face(name: str = Form(...), token: str = Form(...)):
    """[LEGACY] Simpan wajah hasil capture dari kamera pertama."""
    worker = _get_first_camera_worker()
    if not worker:
        raise HTTPException(status_code=404, detail="Belum ada kamera terdaftar di sistem.")
    return await save_face(worker.id, name=name, token=token)


@app.get("/api/status")
async def get_status():
    cams_status = {}
    for cam_id, worker in camera_manager.all().items():
        cams_status[cam_id] = {
            "name": worker.name,
            "status": worker.status,
            "ai_settings": worker.ai_settings,
        }
    return {
        "total_cameras": len(camera_manager.all()),
        "cameras": cams_status,
        "registered_faces": len(set(id_to_name.values())),
        "face_engine": {
            "yunet": face_engine.yunet_available,
            "sface": face_engine.sface_available,
            "samples": face_train_stats.get("samples", 0),
        },
        "legacy_migration": {
            "env_rtsp_url_detected": bool(LEGACY_RTSP_URL),
            "note": "Endpoint /api/toggle, /api/alerts, /video_feed, /api/register/* "
                    "(tanpa cam_id) tetap aktif dan menunjuk ke kamera pertama "
                    "untuk kompatibilitas dengan server.py versi lama.",
        },
        "nx": {
            "server": NX_BASE_URL,
            "plugin_id": NX_PLUGIN_ID or "(tidak diset)",
            "configured": bool(NX_PASSWORD),
            "reachable": nx_status["reachable"],
            "last_error": nx_status["last_error"],
            "since": nx_status["since"],
            "last_checked_at": nx_status["last_checked_at"],
            "poll_interval_sec": NX_POLL_INTERVAL_SEC,
        },
    }


@app.get("/api/stats/system")
async def stats_system():
    """Snapshot kinerja real-time saat ini: CPU, RAM, GPU (jika ada), FPS tiap kamera AI."""
    with _stats_lock:
        return _collect_system_snapshot()


@app.get("/api/stats/history")
async def stats_history(limit: int = 60):
    """
    Histori beberapa menit terakhir untuk grafik dashboard (CPU/RAM/AI FPS over time).
    `limit` = jumlah sample terakhir yang mau diambil (max STATS_HISTORY_LEN).
    """
    with _stats_lock:
        data = list(_stats_history)[-limit:]
    return {
        "poll_interval_sec": STATS_POLL_INTERVAL_SEC,
        "count": len(data),
        "samples": [
            {
                "ts": s["ts"],
                "cpu_pct": s["cpu"]["total_pct"],
                "mem_pct": s["memory"]["pct"],
                "proc_cpu_pct": s["process"]["cpu_pct"],
                "total_fps": s["ai"]["total_fps"],
                "active_ai_cameras": s["ai"]["active_ai_cameras"],
            }
            for s in data
        ],
    }


@app.get("/api/system/network")
async def system_network():
    """
    Info konektivitas untuk membantu setup akses jarak jauh via Tailscale:
    hostname lokal, semua IP yang ke-bind di mesin ini (termasuk IP tailscale 100.x.x.x
    kalau tailscale sedang aktif), dan port server ini berjalan.
    """
    hostname = socket.gethostname()
    ip_list = []
    try:
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ip not in ip_list and not ip.startswith("127."):
                ip_list.append(ip)
    except Exception:
        pass

    tailscale_ip = next((ip for ip in ip_list if ip.startswith("100.")), None)

    return {
        "hostname": hostname,
        "detected_ips": ip_list,
        "tailscale_ip": tailscale_ip,
        "tailscale_detected": bool(tailscale_ip),
        "port": 8000,
        "hint": (
            "Kalau tailscale_detected true, pakai tailscale_ip (atau MagicDNS "
            "'<hostname>.<tailnet>.ts.net') di app mobile sebagai Server Host — "
            "bisa diakses dari luar jaringan lokal tanpa buka port di router."
        ),
    }


@app.get("/api/nx/status")
async def nx_status_live():
    """
    Endpoint RINGAN khusus status reachability NX, didesain untuk
    dipolling dashboard tiap 2-3 detik tanpa overhead query kamera dll.
    """
    with nx_status_lock:
        return {
            "reachable": nx_status["reachable"],
            "last_error": nx_status["last_error"],
            "since": nx_status["since"],
            "last_checked_at": nx_status["last_checked_at"],
            "poll_interval_sec": NX_POLL_INTERVAL_SEC,
        }


# =========================================================
# 🔔 NX WEBHOOK
# =========================================================

@app.post("/nx/webhook")
async def nx_webhook(request: Request):
    # Proteksi endpoint webhook: NX harus mengirim header X-Webhook-Secret yang cocok.
    # Set NX_WEBHOOK_SECRET di env, lalu konfigurasikan header yang sama di sisi NX Optic
    # (Generic Event / HTTP action -> custom header).
    if NX_WEBHOOK_SECRET:
        incoming_secret = request.headers.get("X-Webhook-Secret", "")
        if not secrets.compare_digest(incoming_secret, NX_WEBHOOK_SECRET):
            return JSONResponse(status_code=401, content={"status": "error", "detail": "Invalid webhook secret"})

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    print(f"[NX Webhook] Diterima: {json.dumps(payload, indent=2)}")
    event_type = payload.get("eventType", "")
    device_id = payload.get("deviceId", "")

    # Cari kamera yang nx_camera_id-nya cocok, lalu aktifkan face recognition
    for cam_id, worker in camera_manager.all().items():
        if worker.nx_camera_id == device_id and event_type == "cameraMotionEvent":
            worker.update_ai_settings(face_recognition=True)
            print(f"[NX Webhook] Motion di '{worker.name}' → face recognition diaktifkan.")

    return JSONResponse({"status": "received", "event": event_type})


# =========================================================
# 🧪 NX TEST & DIAGNOSE
# =========================================================

@app.get("/api/nx/test")
async def nx_test():
    try:
        token = _nx_get_token()
        if not token:
            return JSONResponse(status_code=401, content={
                "status": "error", "detail": "Gagal login ke NX. Cek NX_USER/NX_PASSWORD.",
            })
        resp = requests.get(f"{NX_BASE_URL}/rest/v3/system/info", headers=_nx_headers(), timeout=5, verify=False)
        if resp.ok:
            info = resp.json()
            # NX kadang membungkus response dalam {"reply": {...}} tergantung versi/endpoint
            if isinstance(info, dict) and "reply" in info and isinstance(info["reply"], dict):
                info = info["reply"]
            nx_name = info.get("systemName") or info.get("name") or info.get("localSystemName")
            nx_ver  = info.get("version") or info.get("serverVersion")
            kirim_event_ke_nx("🧪 Test Koneksi AI VMS Server", "Koneksi NX v6 berhasil.")
            return {"status": "ok", "nx_name": nx_name, "nx_ver": nx_ver, "raw_keys": list(info.keys())}
        return JSONResponse(status_code=502, content={"status": "error", "http": resp.status_code, "detail": resp.text[:300]})
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.get("/api/nx/cameras")
async def nx_cameras():
    """Daftar kamera yang terdaftar di NX Optic (untuk mapping nx_camera_id saat add camera)."""
    try:
        resp = requests.get(f"{NX_BASE_URL}/rest/v3/devices", headers=_nx_headers(), timeout=5, verify=False)
        if resp.ok:
            cameras = [
                {"id": c.get("id"), "name": c.get("name"), "model": c.get("model"), "url": c.get("url")}
                for c in resp.json() if c.get("deviceType") == "camera"
            ]
            return {"cameras": cameras, "total": len(cameras)}
        return JSONResponse(status_code=502, content={"status": "error", "code": resp.status_code})
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.get("/api/nx/diagnose")
async def nx_diagnose():
    result = {"config": {
        "nx_base_url": NX_BASE_URL, "nx_user": NX_USER, "has_password": bool(NX_PASSWORD),
    }, "steps": {}}

    for scheme in ["https", "http"]:
        base = f"{scheme}://{NX_SERVER_IP}:{NX_PORT}"
        try:
            r = requests.get(f"{base}/rest/v3/system/info", timeout=4, verify=False)
            result["steps"][f"ping_{scheme}"] = {"ok": True, "status": r.status_code}
        except Exception as e:
            result["steps"][f"ping_{scheme}"] = {"ok": False, "error": str(e)}

    _nx_token_cache["token"] = None
    token = _nx_get_token()
    result["steps"]["login"] = {"ok": bool(token), "error": nx_status.get("last_error") if not token else None}
    if not token:
        result["verdict"] = "❌ Gagal login. Cek NX_USER/NX_PASSWORD/IP."
        return JSONResponse(status_code=401, content=result)

    try:
        r = requests.get(f"{NX_BASE_URL}/rest/v3/devices", headers=_nx_headers(), timeout=5, verify=False)
        if r.ok:
            cams = [d for d in r.json() if d.get("deviceType") == "camera"]
            result["steps"]["devices"] = {"ok": True, "total_cameras": len(cams), "camera_ids": [c.get("id") for c in cams][:10]}
    except Exception as e:
        result["steps"]["devices"] = {"ok": False, "error": str(e)}

    all_ok = all(v.get("ok") for v in result["steps"].values())
    result["verdict"] = "✅ Semua koneksi NX OK!" if all_ok else "⚠️ Ada langkah yang gagal."
    return result


@app.post("/api/nx/bookmark")
async def nx_bookmark_manual(name: str, nx_camera_id: str, description: str = "", tags: str = "manual"):
    buat_bookmark_nx(name, description, nx_camera_id, tags=tags.split(","))
    return {"status": "ok", "bookmark": name}


# =========================================================
# 🔁 NX REACHABILITY POLLER (real-time-ish status monitor)
# =========================================================
# NX Optic tidak punya mekanisme push-event untuk status server-nya
# sendiri (online/offline), jadi kita polling cepat (default 3 detik)
# dan hanya trigger notifikasi saat status BERUBAH — bukan tiap poll.

def _check_nx_reachable_now() -> tuple[bool, Optional[str]]:
    """Cek cepat apakah NX bisa diakses & token masih valid. Tidak pakai cache token lama."""
    if not NX_SERVER_IP or not NX_USER or not NX_PASSWORD:
        return False, "NX belum dikonfigurasi (server IP/user/password kosong)"
    try:
        resp = requests.get(
            f"{NX_BASE_URL}/rest/v3/system/info",
            headers=_nx_headers(), timeout=NX_POLL_INTERVAL_SEC + 1, verify=False,
        )
        if resp.status_code == 401:
            # Token expired, paksa refresh sekali lalu coba ulang
            _nx_token_cache["token"] = None
            resp = requests.get(
                f"{NX_BASE_URL}/rest/v3/system/info",
                headers=_nx_headers(), timeout=NX_POLL_INTERVAL_SEC + 1, verify=False,
            )
        if resp.ok:
            return True, None
        return False, f"HTTP {resp.status_code}: {resp.text[:150]}"
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {e}"
    except requests.exceptions.Timeout:
        return False, "Timeout — NX tidak merespons"
    except Exception as e:
        return False, str(e)


def _nx_reachability_poller():
    """
    Loop tanpa henti, polling tiap NX_POLL_INTERVAL_SEC detik.
    Begitu status reachable berubah (True<->False), langsung:
    1. Update nx_status dengan timestamp perubahan
    2. Print log perubahan
    3. Kalau NX baru ONLINE lagi (reconnect) → kirim Generic Event ke NX
       sebagai catatan recovery (offline tidak bisa dikirimi event, jelas).
    """
    time.sleep(2)  # beri waktu FastAPI siap sebelum mulai polling
    prev_reachable = None  # None = belum pernah dicek

    while True:
        reachable, error = _check_nx_reachable_now()
        now = time.time()

        with nx_status_lock:
            nx_status["last_checked_at"] = now
            state_changed = (prev_reachable is not None and prev_reachable != reachable)
            first_check = (prev_reachable is None)

            if state_changed or first_check:
                nx_status["last_changed_at"] = now
                nx_status["since"] = datetime.now().isoformat()

            nx_status["reachable"] = reachable
            nx_status["last_error"] = error

        if state_changed:
            if reachable:
                print(f"[NX Poller ✅] NX Optic ONLINE kembali (sebelumnya offline).")
                # Kirim event recovery ke NX — tercatat di log NX juga
                kirim_event_ke_nx(
                    "🟢 NX Server Reconnected",
                    "Koneksi AI VMS Server ke NX Optic pulih kembali setelah sempat terputus.",
                )
            else:
                print(f"[NX Poller ❌] NX Optic OFFLINE/unreachable. Error: {error}")
                # Tidak bisa kirim event ke NX karena NX sedang unreachable —
                # ini hanya tercatat di log server & banner dashboard.
        elif first_check:
            status_text = "ONLINE ✅" if reachable else f"OFFLINE ❌ ({error})"
            print(f"[NX Poller] Status awal NX: {status_text}")

        prev_reachable = reachable
        time.sleep(NX_POLL_INTERVAL_SEC)


threading.Thread(target=_nx_reachability_poller, daemon=True).start()

# ===== Modul Parking =====
import parking_api
parking_api.init(get_conn, nx_alert=lapor_deteksi_nx)
app.include_router(parking_api.router)

@app.get("/parking", response_class=HTMLResponse)
async def parking_page():
    path = os.path.join(os.path.dirname(__file__), "parking3d.html")
    return HTMLResponse(open(path, encoding="utf-8").read())

# =========================================================
# 🚀 ENTRYPOINT
# =========================================================
enpidix_fleet.install(app, globals())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)