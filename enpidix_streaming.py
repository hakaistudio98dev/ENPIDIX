"""
ENPIDIX — Modul Live Streaming WebRTC
=====================================

Melengkapi server.py: live view H.264 passthrough lewat go2rtc.

Kenapa perlu, padahal MJPEG di server.py sudah dioptimalkan?
  MJPEG mengirim JPEG UTUH tiap frame. Di 720p/15fps itu ~8-15 Mbps — lewat
  4G/Tailscale pasti tersendat, seberapa pun rapinya kode encoder. WebRTC
  meneruskan H.264 ASLI dari kamera (~1-2 Mbps) tanpa decode/encode di server:
  CPU server nyaris 0%, gambar tetap tajam, latensi <1 detik.

Yang TIDAK berubah:
  - CameraWorker, AI (YOLO/LBPH), rekaman ffmpeg, /video_feed lama — semua utuh.
  - Box AI tetap tampil: video WebRTC dilapisi <canvas> yang menggambar box dari
    detections (koordinat sudah dinormalisasi 0-1 oleh _norm_box di server.py).

Kalau go2rtc gagal jalan, semua endpoint otomatis jatuh ke MJPEG — server tetap
berfungsi seperti sebelumnya.

Dependensi: pip install httpx websockets pyyaml
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

try:
    import httpx
except ImportError:
    httpx = None
try:
    import websockets
except ImportError:
    websockets = None

log = logging.getLogger("enpidix.streaming")

BASE_DIR = Path(__file__).resolve().parent
GO2RTC_DIR = BASE_DIR / "bin"
GO2RTC_CONFIG = BASE_DIR / "go2rtc.yaml"

GO2RTC_VERSION = os.environ.get("GO2RTC_VERSION", "v1.9.9")
GO2RTC_API_PORT = int(os.environ.get("GO2RTC_API_PORT", "1984"))
GO2RTC_WEBRTC_PORT = int(os.environ.get("GO2RTC_WEBRTC_PORT", "8555"))
GO2RTC_ENABLED = os.environ.get("ENPIDIX_WEBRTC", "true").lower() != "false"

# Alamat server yang dilihat ponsel (Tailscale 100.x.x.x atau IP LAN).
# WAJIB diisi kalau ponsel mengakses dari luar jaringan lokal — tanpa ini
# ICE candidate WebRTC kosong dan koneksi tidak akan pernah terbentuk.
PUBLIC_HOST = os.environ.get("ENPIDIX_PUBLIC_HOST", "").strip()

GO2RTC_API = f"http://127.0.0.1:{GO2RTC_API_PORT}"

# Diisi lewat configure() dari server.py (menghindari circular import)
_fetch_cameras: Optional[Callable[[], List[dict]]] = None
_get_worker: Optional[Callable[[int], object]] = None
_secret: str = "enpidix"


def configure(fetch_cameras: Callable[[], List[dict]],
              get_worker: Callable[[int], object],
              secret: str) -> None:
    global _fetch_cameras, _get_worker, _secret
    _fetch_cameras = fetch_cameras
    _get_worker = get_worker
    _secret = secret or "enpidix"


# =========================================================
# 🔑 Token stream
# =========================================================
# PENTING: BaseHTTPMiddleware di Starlette hanya memproses request HTTP —
# koneksi WebSocket LOLOS tanpa diperiksa AuthMiddleware. Jadi signaling
# WebRTC diamankan token HMAC berumur pendek yang hanya bisa diambil lewat
# /api/stream/token (endpoint itu tetap dijaga AuthMiddleware seperti biasa).

TOKEN_TTL_SEC = int(os.environ.get("ENPIDIX_STREAM_TOKEN_TTL", "43200"))  # 12 jam
SYNC_INTERVAL_SEC = int(os.environ.get("ENPIDIX_STREAM_SYNC_SEC", "20"))  # cek DB tiap 20 detik


def issue_stream_token(ttl: int = TOKEN_TTL_SEC) -> str:
    exp = int(time.time()) + ttl
    sig = hmac.new(_secret.encode(), str(exp).encode(), hashlib.sha256).hexdigest()[:32]
    return f"{exp}.{sig}"


def verify_stream_token(token: str) -> bool:
    if not token or "." not in token:
        return False
    exp_str, _, sig = token.partition(".")
    try:
        exp = int(exp_str)
    except ValueError:
        return False
    if exp < time.time():
        return False
    expected = hmac.new(_secret.encode(), exp_str.encode(), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)


# =========================================================
# 🎯 Sub-stream
# =========================================================
# Sub-stream (D1/480p) untuk grid + AI, main stream (1080p) untuk fullscreen.
# Ini penghematan CPU terbesar: 8 kamera di grid tidak lagi menarik 8x 1080p,
# dan YOLO tidak lagi men-decode 1080p hanya untuk di-resize ke 320px.

_SUBSTREAM_RULES = [
    (re.compile(r"(/Streaming/Channels/\d)01\b"), r"\g<1>02"),   # Hikvision
    (re.compile(r"/main/"), "/sub/"),                            # Hikvision lama
    (re.compile(r"([?&]subtype=)0\b"), r"\g<1>1"),                # Dahua / Imou
    (re.compile(r"(/media/video)1\b"), r"\g<1>2"),                # Uniview
    (re.compile(r"(/stream)1\b"), r"\g<1>2"),                     # generik
    (re.compile(r"(/cam/realmonitor\?channel=\d+&subtype=)0"), r"\g<1>1"),
]


def guess_substream_url(rtsp_url: str) -> Optional[str]:
    """Tebak URL sub-stream dari URL main stream. None kalau pola tak dikenali."""
    if not rtsp_url:
        return None
    for pattern, repl in _SUBSTREAM_RULES:
        if pattern.search(rtsp_url):
            return pattern.sub(repl, rtsp_url, count=1)
    return None


def resolve_substream(row: dict) -> Optional[str]:
    """Pakai kolom rtsp_url_sub kalau diisi admin; kalau kosong, coba tebak."""
    try:
        manual = (row.get("rtsp_url_sub") or "").strip()
    except AttributeError:
        manual = ""
    if manual:
        return manual
    return guess_substream_url(row.get("rtsp_url", ""))


# =========================================================
# 🚀 Manager go2rtc
# =========================================================

class Go2RtcManager:
    """Menjalankan go2rtc sebagai proses anak dari server.py."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._streams: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._watcher: Optional[threading.Thread] = None

    # -- sinkronisasi dari DB ---------------------------------------------

    def sync_from_db(self) -> None:
        """Baca ulang kamera dari DB, tulis go2rtc.yaml, reload kalau berubah."""
        if _fetch_cameras is None:
            return
        try:
            rows = _fetch_cameras()
        except Exception as exc:
            log.warning("Gagal membaca kamera dari DB: %s", exc)
            return

        streams: Dict[str, str] = {}
        for row in rows:
            cam_id = row["id"]
            main = (row.get("rtsp_url") or "").strip()
            if not main:
                continue
            streams[f"cam{cam_id}"] = main
            sub = resolve_substream(row)
            if sub and sub != main:
                streams[f"cam{cam_id}_sub"] = sub

        with self._lock:
            changed = streams != self._streams
            self._streams = streams

        if changed:
            self._write_config()
            if self.running:
                self.reload()

    def stream_name(self, cam_id: int, profile: str = "sub") -> Optional[str]:
        with self._lock:
            main_key, sub_key = f"cam{cam_id}", f"cam{cam_id}_sub"
            if profile == "sub" and sub_key in self._streams:
                return sub_key
            return main_key if main_key in self._streams else None

    def rtsp_for_analytics(self, cam_id: int) -> Optional[str]:
        """URL yang sebaiknya dipakai CameraWorker (sub-stream kalau tersedia)."""
        with self._lock:
            return self._streams.get(f"cam{cam_id}_sub") or self._streams.get(f"cam{cam_id}")

    def restream_url(self, cam_id: int, profile: str = "sub") -> Optional[str]:
        """
        URL RTSP LOKAL (127.0.0.1:8554) yang diteruskan go2rtc.

        Kenapa ini penting: tanpa konsolidasi ini, satu kamera bisa dibuka
        RTSP-nya 3-4 kali sekaligus — oleh CameraWorker (AI), CameraRecorder
        (ffmpeg rekaman), dan go2rtc sendiri (WebRTC live view). Banyak kamera
        CCTV murah (klon Hikvision/Dahua) hanya mendukung 2-3 koneksi RTSP
        BERSAMAAN; lebih dari itu koneksi mulai ditolak atau stream jadi
        corrupt. Dengan restream_url(), go2rtc menarik H.264 dari kamera
        SATU KALI per profile (main/sub), lalu semua konsumer internal
        (AI, rekaman, WebRTC) mengambil dari go2rtc — bukan dari kamera.

        Return None kalau go2rtc belum siap -> caller WAJIB fallback ke URL
        RTSP langsung ke kamera (semua caller di server.py sudah begitu).
        """
        if not self.running:
            return None
        name = self.stream_name(cam_id, profile)
        if not name:
            return None
        return f"rtsp://127.0.0.1:8554/{name}"

    def wait_ready(self, timeout: float = 6.0) -> bool:
        """
        Tunggu sebentar sampai HTTP API go2rtc merespons (menandakan proses
        sudah hidup dan RTSP listener-nya siap menerima koneksi). Dipanggil
        SEKALI saat startup server, sebelum kamera dimuat dari DB — supaya
        CameraWorker/CameraRecorder tidak langsung gagal konek ke go2rtc yang
        masih dalam proses booting. Best-effort: kalau timeout, caller tetap
        lanjut — retry per-kamera di CameraWorker akan menutupi sisanya.
        """
        if not self.running or httpx is None:
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{GO2RTC_API}/api/streams", timeout=1.5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.4)
        return False

    # -- config ------------------------------------------------------------

    def _write_config(self) -> None:
        with self._lock:
            # #backchannel=0 mencegah go2rtc menegosiasi audio dua arah yang
            # tidak didukung banyak kamera dan bikin stream gagal terbuka.
            streams = {name: [f"{url}#backchannel=0"] for name, url in self._streams.items()}

        candidates = [f"{PUBLIC_HOST}:{GO2RTC_WEBRTC_PORT}"] if PUBLIC_HOST else []
        cfg = {
            "streams": streams,
            # API hanya di localhost — akses luar lewat proxy FastAPI yang ber-auth
            "api": {"listen": f"127.0.0.1:{GO2RTC_API_PORT}", "origin": "*"},
            "webrtc": {"listen": f":{GO2RTC_WEBRTC_PORT}", "candidates": candidates},
            "rtsp": {"listen": "127.0.0.1:8554"},
            "ffmpeg": {"bin": shutil.which("ffmpeg") or "ffmpeg"},
            "log": {"level": "warn"},
        }
        try:
            GO2RTC_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            log.warning("Gagal menulis go2rtc.yaml: %s", exc)

    # -- binary ------------------------------------------------------------

    def _binary_path(self) -> Path:
        exe = "go2rtc.exe" if platform.system() == "Windows" else "go2rtc"
        found = shutil.which(exe)
        return Path(found) if found else GO2RTC_DIR / exe

    def _download_binary(self) -> Path:
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "windows":
            asset = "go2rtc_win64.zip"
        elif system == "darwin":
            asset = "go2rtc_mac_arm64.zip" if "arm" in machine else "go2rtc_mac_amd64.zip"
        elif "aarch64" in machine or "arm64" in machine:
            asset = "go2rtc_linux_arm64"
        else:
            asset = "go2rtc_linux_amd64"

        url = f"https://github.com/AlexxIT/go2rtc/releases/download/{GO2RTC_VERSION}/{asset}"
        GO2RTC_DIR.mkdir(parents=True, exist_ok=True)
        target = GO2RTC_DIR / ("go2rtc.exe" if system == "windows" else "go2rtc")

        print(f"[go2rtc ⬇️ ] Mengunduh {asset} (~15 MB, sekali saja)...")
        tmp = GO2RTC_DIR / asset
        urllib.request.urlretrieve(url, tmp)

        if asset.endswith(".zip"):
            with zipfile.ZipFile(tmp) as z:
                z.extractall(GO2RTC_DIR)
            tmp.unlink(missing_ok=True)
        else:
            shutil.move(str(tmp), str(target))

        if system != "windows":
            target.chmod(target.stat().st_mode | stat.S_IEXEC)
        return target

    # -- siklus hidup ------------------------------------------------------

    def start(self) -> None:
        if not GO2RTC_ENABLED:
            self._last_error = "dinonaktifkan (ENPIDIX_WEBRTC=false)"
            print("[go2rtc ⏸️ ] Dinonaktifkan — live view memakai MJPEG.")
            return
        if self.running:
            return
        if not (httpx and websockets):
            self._last_error = "modul httpx/websockets belum terpasang"
            print("[go2rtc ⚠️ ] Jalankan: pip install httpx websockets pyyaml")
            return

        self.sync_from_db()
        if not GO2RTC_CONFIG.exists():
            self._write_config()

        binary = self._binary_path()
        if not binary.exists():
            try:
                binary = self._download_binary()
            except Exception as exc:
                self._last_error = f"unduh gagal: {exc}"
                print(f"[go2rtc ⚠️ ] Gagal mengunduh ({exc}). Live view jatuh ke MJPEG.")
                return

        flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        try:
            self._proc = subprocess.Popen(
                [str(binary), "-config", str(GO2RTC_CONFIG)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[go2rtc ⚠️ ] Gagal menjalankan ({exc}). Live view jatuh ke MJPEG.")
            return

        self._last_error = None
        self._start_watcher()
        host_note = PUBLIC_HOST or "⚠️ ENPIDIX_PUBLIC_HOST belum diset — akses luar LAN akan gagal"
        print(f"[go2rtc ✅] Berjalan (pid={self._proc.pid}) — WebRTC {host_note}:{GO2RTC_WEBRTC_PORT}")

    # -- watcher -----------------------------------------------------------

    def _watch_loop(self) -> None:
        """
        Cek DB berkala supaya penambahan/penghapusan/penonaktifan kamera
        lewat dashboard langsung tercermin di go2rtc — tanpa perlu menempel
        pemanggilan sync di setiap endpoint CRUD (dan tanpa risiko terlewat
        kalau nanti ada endpoint baru).
        """
        while True:
            time.sleep(SYNC_INTERVAL_SEC)
            if not self.running:
                continue
            try:
                self.sync_from_db()
            except Exception as exc:
                log.warning("Auto-sync go2rtc gagal: %s", exc)

    def _start_watcher(self) -> None:
        if self._watcher is None:
            self._watcher = threading.Thread(
                target=self._watch_loop, name="go2rtc-watch", daemon=True)
            self._watcher.start()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print("[go2rtc 🛑] Dihentikan.")
        self._proc = None

    def reload(self) -> None:
        """go2rtc memuat ulang config lewat API, tanpa restart proses."""
        if httpx is None:
            return
        try:
            httpx.post(f"{GO2RTC_API}/api/restart", timeout=3)
        except Exception:
            pass

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        with self._lock:
            names = sorted(self._streams.keys())
        return {
            "enabled": GO2RTC_ENABLED,
            "running": self.running,
            "mode": "webrtc" if self.running else "mjpeg",
            "public_host": PUBLIC_HOST or None,
            "webrtc_port": GO2RTC_WEBRTC_PORT,
            "streams": names,
            "deps_ok": bool(httpx and websockets),
            "error": self._last_error,
        }


go2rtc = Go2RtcManager()


# =========================================================
# 🌐 Router
# =========================================================

streaming_router = APIRouter(prefix="/api/stream", tags=["stream"])


@streaming_router.get("/status")
async def stream_status():
    return go2rtc.status()


@streaming_router.get("/token")
async def stream_token():
    """
    Dipanggil app Android (dengan Basic Auth) sesudah login.
    Token dipakai WebView & WebSocket signaling yang tidak bisa membawa
    header Authorization sendiri.
    """
    return {"token": issue_stream_token(), "expires_in": TOKEN_TTL_SEC}


@streaming_router.post("/sync")
async def stream_sync():
    """Panggil setelah kamera ditambah/dihapus/diedit."""
    go2rtc.sync_from_db()
    if not go2rtc.running:
        go2rtc.start()
    return go2rtc.status()


@streaming_router.get("/{cam_id}/detections")
async def stream_detections(cam_id: int, token: str = Query("")):
    """Isi sama dengan /api/cameras/{id}/detections, tapi bisa diakses dengan
    token stream — dipakai sebagai FALLBACK kalau WebSocket (di bawah) gagal
    connect. Jalur utama overlay sekarang lewat /detections/ws."""
    worker = _get_worker(cam_id) if _get_worker else None
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")
    with worker.lock:
        boxes = list(worker.latest_detections)
    return {"camera_id": cam_id, "ts": time.time(), "boxes": boxes}


@streaming_router.websocket("/{cam_id}/detections/ws")
async def stream_detections_ws(ws: WebSocket, cam_id: int, token: str = Query("")):
    """
    Push box deteksi AI ke overlay lewat WebSocket, bukan di-poll client tiap
    300ms lewat HTTP terpisah.

    Kenapa ini yang bikin overlay terasa "patah-patah" sebelumnya: tiap poll
    HTTP kena overhead koneksi baru + request/response penuh (bisa puluhan-
    ratusan ms tambahan tergantung jaringan), dan update box HANYA muncul
    setiap 300ms sekali — terlihat "melompat" dibanding video yang jalan di
    15-30fps. Di sini server membaca latest_detections tiap ~80ms (worker.lock
    ringan, bukan I/O) dan HANYA mengirim kalau isinya berubah -- update jadi
    jauh lebih rapat DAN lebih murah (satu koneksi persisten, bukan request
    baru tiap kali). Sisi client lalu menginterpolasi antar update lewat
    requestAnimationFrame supaya box meluncur, bukan melompat -- lihat
    _OVERLAY_JS.
    """
    if not verify_stream_token(token):
        await ws.close(code=4401)
        return
    worker = _get_worker(cam_id) if _get_worker else None
    if not worker:
        await ws.close(code=4404)
        return

    await ws.accept()
    last_sent: Optional[list] = None
    try:
        while True:
            with worker.lock:
                boxes = list(worker.latest_detections)
            if boxes != last_sent:
                await ws.send_json({"boxes": boxes, "ts": time.time()})
                last_sent = boxes
            await asyncio.sleep(0.08)  # ~12x/detik -- cukup halus, jauh lebih murah dari polling HTTP
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("Detections WS gagal untuk cam %s: %s", cam_id, exc)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@streaming_router.get("/{cam_id}/mjpeg")
async def stream_mjpeg(cam_id: int, token: str = Query("")):
    """Fallback MJPEG — memakai cache JPEG CameraWorker, tanpa encode ulang."""
    worker = _get_worker(cam_id) if _get_worker else None
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")

    async def gen():
        last_version = -1
        while True:
            jpeg, version = worker.get_jpeg_with_version()
            if jpeg is not None and version != last_version:
                last_version = version
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@streaming_router.get("/{cam_id}/player", response_class=HTMLResponse)
async def player(cam_id: int,
                 profile: str = Query("sub", pattern="^(main|sub)$"),
                 overlay: int = Query(1),
                 token: str = Query("")):
    """
    Halaman pemutar untuk WebView app Android / dashboard.
    WebRTC kalau go2rtc hidup, MJPEG kalau tidak — klien tidak perlu tahu bedanya.

      profile=sub   -> hemat, untuk grid
      profile=main  -> resolusi penuh, untuk fullscreen
      overlay=0     -> matikan box AI (video bersih)
    """
    worker = _get_worker(cam_id) if _get_worker else None
    if not worker:
        raise HTTPException(status_code=404, detail="Kamera tidak ditemukan/aktif")

    tok = token if verify_stream_token(token) else issue_stream_token()
    src = go2rtc.stream_name(cam_id, profile) if go2rtc.running else None

    if src:
        return HTMLResponse(_WEBRTC_PAGE.format(
            src=src, cam_id=cam_id, token=tok, overlay=int(bool(overlay))))
    return HTMLResponse(_MJPEG_PAGE.format(
        cam_id=cam_id, token=tok, overlay=int(bool(overlay))))


@streaming_router.websocket("/ws")
async def go2rtc_ws_proxy(ws: WebSocket, src: str = Query(...), token: str = Query("")):
    """
    Proxy signaling WebRTC ke go2rtc. Karena lewat sini, port 1984 tetap
    tertutup dari jaringan luar dan hanya port 8000 yang perlu dijangkau.
    """
    if not verify_stream_token(token):
        await ws.close(code=4401)
        return
    if websockets is None:
        await ws.close(code=4503)
        return

    await ws.accept()
    upstream_url = f"ws://127.0.0.1:{GO2RTC_API_PORT}/api/ws?src={src}"
    try:
        async with websockets.connect(upstream_url, max_size=None) as upstream:

            async def to_upstream():
                while True:
                    await upstream.send(await ws.receive_text())

            async def to_client():
                async for msg in upstream:
                    await ws.send_text(msg if isinstance(msg, str) else msg.decode())

            tasks = [asyncio.create_task(to_upstream()), asyncio.create_task(to_client())]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("Proxy WS gagal untuk %s: %s", src, exc)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# =========================================================
# 📺 Halaman pemutar
# =========================================================

_OVERLAY_JS = """
const camId = {cam_id}, token = "{token}", useOverlay = {overlay};
const cv = document.getElementById('ov');
const ctx = cv.getContext('2d');
const COLORS = {{ face:'#02CBE2', smoking:'#FFB020', fire:'#FF4757',
                 behavior:'#B5ABFC', count:'#9184D9', jatuh:'#FF4757' }};
function colorOf(t) {{
  for (const k in COLORS) if (t && t.indexOf(k) === 0) return COLORS[k];
  return '#9184D9';
}}
function draw(boxes, el) {{
  const r = el.getBoundingClientRect();
  if (cv.width !== r.width || cv.height !== r.height) {{ cv.width = r.width; cv.height = r.height; }}
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.lineWidth = 2; ctx.font = '12px system-ui';
  for (const b of boxes) {{
    const x = b.x1*cv.width, y = b.y1*cv.height;
    const w = (b.x2-b.x1)*cv.width, h = (b.y2-b.y1)*cv.height;
    const c = colorOf(b.type);
    ctx.strokeStyle = c; ctx.strokeRect(x, y, w, h);
    if (b.label) {{
      const tw = ctx.measureText(b.label).width + 8;
      ctx.fillStyle = c; ctx.fillRect(x, Math.max(0, y-16), tw, 16);
      ctx.fillStyle = '#0b0d14'; ctx.fillText(b.label, x+4, Math.max(12, y-4));
    }}
  }}
}}

// --- Box AI mengalir mulus, bukan melompat tiap update ---
// targetBoxes = posisi TERBARU dari server (lewat WebSocket, bukan polling
// fetch tiap 300ms lagi -- itu penyebab utama box terasa patah-patah:
// overhead request baru tiap poll + update cuma 3x/detik, jauh di bawah
// framerate video). renderBoxes = posisi yang BENAR-BENAR digambar, digeser
// sedikit demi sedikit ke arah target tiap frame lewat requestAnimationFrame
// (selaras refresh rate layar, ~60fps) -- box jadi meluncur halus mengikuti
// gerakan objek, walau data baru dari server hanya datang ~12x/detik.
let targetBoxes = [];
let renderBoxes = [];
let detWs = null;
let pollTimer = null;

function lerp(a, b, t) {{ return a + (b - a) * t; }}

function matchBox(target, prevList) {{
  // Cocokkan box baru dengan box lama berdasarkan tipe + posisi terdekat,
  // supaya interpolasi tidak "meluncur" antar objek yang berbeda saat ada
  // beberapa box dengan tipe sama (mis. dua wajah) dalam satu frame.
  let best = null, bestDist = 0.15; // ambang: kalau lompatan > ini, anggap objek baru, jangan diinterpolasi
  for (const p of prevList) {{
    if (p.type !== target.type) continue;
    const dist = Math.abs(p.x1 - target.x1) + Math.abs(p.y1 - target.y1);
    if (dist < bestDist) {{ bestDist = dist; best = p; }}
  }}
  return best;
}}

function stepInterpolation() {{
  const next = targetBoxes.map(t => {{
    const prev = matchBox(t, renderBoxes);
    if (!prev) return t; // objek baru muncul -- langsung tampil, tidak ada "dari mana" untuk diinterpolasi
    return {{
      type: t.type, label: t.label,
      x1: lerp(prev.x1, t.x1, 0.35), y1: lerp(prev.y1, t.y1, 0.35),
      x2: lerp(prev.x2, t.x2, 0.35), y2: lerp(prev.y2, t.y2, 0.35),
    }};
  }});
  renderBoxes = next;
}}

function renderLoop(el) {{
  stepInterpolation();
  draw(renderBoxes, el);
  requestAnimationFrame(() => renderLoop(el));
}}

function startPolling() {{
  if (pollTimer) return;
  pollTimer = setInterval(async () => {{
    if (document.hidden) return;
    try {{
      const res = await fetch('/api/stream/' + camId + '/detections?token=' + token);
      const d = await res.json();
      targetBoxes = d.boxes || [];
    }} catch (e) {{}}
  }}, 300);
}}

function connectDetectionsWs() {{
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  detWs = new WebSocket(proto + '//' + location.host + '/api/stream/' + camId + '/detections/ws?token=' + token);
  detWs.onmessage = (ev) => {{
    try {{
      const d = JSON.parse(ev.data);
      targetBoxes = d.boxes || [];
    }} catch (e) {{}}
  }};
  detWs.onerror = () => {{ startPolling(); }}; // WebSocket gagal (proxy lama, dll) -> tetap jalan lewat polling
  detWs.onclose = () => {{ setTimeout(() => {{ if (!document.hidden) connectDetectionsWs(); }}, 1500); }};
}}

function startOverlay(el) {{
  if (!useOverlay) return;
  connectDetectionsWs();
  requestAnimationFrame(() => renderLoop(el));
}}
"""

_WEBRTC_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
 html,body{{margin:0;height:100%;background:#000;overflow:hidden}}
 #wrap{{position:relative;width:100%;height:100%}}
 video,#ov{{position:absolute;inset:0;width:100%;height:100%}}
 video{{object-fit:contain;background:#000}}
 #ov{{pointer-events:none}}
 #msg{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
      color:#B5ABFC;font:13px system-ui;background:rgba(0,0,0,.55)}}
</style></head><body>
<div id="wrap">
  <video id="v" autoplay playsinline muted></video>
  <canvas id="ov"></canvas>
  <div id="msg">Menyambung…</div>
</div>
<script>
""" + _OVERLAY_JS + """
const src = "{src}";
const video = document.getElementById('v');
const msg = document.getElementById('msg');
let pc = null, ws = null, retry = 0;

function cleanup() {{
  if (pc) {{ try {{ pc.close(); }} catch(e) {{}} pc = null; }}
  if (ws) {{ try {{ ws.close(); }} catch(e) {{}} ws = null; }}
}}
function scheduleRetry() {{
  cleanup();
  msg.style.display = 'flex';
  retry = Math.min(retry + 1, 6);
  setTimeout(connect, retry * 1200);
}}
function connect() {{
  cleanup();
  msg.style.display = 'flex';
  pc = new RTCPeerConnection({{ iceServers: [], bundlePolicy: 'max-bundle' }});
  pc.addTransceiver('video', {{ direction: 'recvonly' }});
  pc.addTransceiver('audio', {{ direction: 'recvonly' }});
  pc.ontrack = e => {{ video.srcObject = e.streams[0]; msg.style.display = 'none'; retry = 0; }};
  pc.onconnectionstatechange = () => {{
    if (['failed','disconnected','closed'].indexOf(pc.connectionState) >= 0) scheduleRetry();
  }};

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host +
      '/api/stream/ws?src=' + encodeURIComponent(src) + '&token=' + token);
  ws.onopen = async () => {{
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    ws.send(JSON.stringify({{ type:'webrtc/offer', value: offer.sdp }}));
  }};
  ws.onmessage = async ev => {{
    const m = JSON.parse(ev.data);
    if (m.type === 'webrtc/answer') {{
      await pc.setRemoteDescription({{ type:'answer', sdp: m.value }});
    }} else if (m.type === 'webrtc/candidate' && m.value) {{
      try {{ await pc.addIceCandidate({{ candidate: m.value, sdpMid: '0' }}); }} catch(e) {{}}
    }} else if (m.type === 'error') {{
      msg.textContent = 'Stream tidak tersedia';
    }}
  }};
  ws.onclose = () => {{ if (!video.srcObject) scheduleRetry(); }};
}}

connect();
startOverlay(video);

// Hemat baterai & kuota: putus saat app di background, sambung lagi saat kembali.
document.addEventListener('visibilitychange', () => {{
  if (document.hidden) cleanup();
  else if (!pc) connect();
}});
</script></body></html>"""

_MJPEG_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
 html,body{{margin:0;height:100%;background:#000;overflow:hidden}}
 #wrap{{position:relative;width:100%;height:100%}}
 img,#ov{{position:absolute;inset:0;width:100%;height:100%}}
 img{{object-fit:contain}} #ov{{pointer-events:none}}
</style></head><body>
<div id="wrap">
  <img id="v" src="/api/stream/{cam_id}/mjpeg?token={token}">
  <canvas id="ov"></canvas>
</div>
<script>
""" + _OVERLAY_JS + """
startOverlay(document.getElementById('v'));
</script></body></html>"""