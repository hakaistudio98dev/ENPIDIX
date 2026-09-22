"""
ENPIDIX Fleet — modul penyatu
=============================
Menjadikan `server.py` yang sudah ada berjalan sebagai **Edge Node** dalam
arsitektur 3 tier, tanpa menulis ulang 4.000+ baris yang sudah bekerja.

Yang ditambahkan:
  1. Identitas fleet (engine_id / site_id) + pendaftaran ke regional
  2. Store & forward — event lokal tetap aman saat WAN putus
  3. Heartbeat dengan telemetri Jetson (tegrastats)
  4. Tiga patch performa yang bisa dinyalakan satu per satu dan diukur

Pasang di server.py dengan TIGA baris (pola yang sama seperti modul parking):

    # 1) di dekat import lain, setelah `app = FastAPI(...)`
    import enpidix_fleet

    # 2) tepat sebelum blok `if __name__ == "__main__":`
    enpidix_fleet.install(app, globals())

    # 3) (opsional) di dalam handler shutdown yang sudah ada
    enpidix_fleet.shutdown()

Tentang patch performa
----------------------
Patch TIDAK menyala otomatis. Masing-masing punya saklar sendiri di dashboard
(tab Settings → Fleet) karena efeknya berbeda-beda per mesin, dan Anda perlu
bisa mengukur sebelum/sesudah. Nyalakan satu per satu, lihat FPS dan CPU-nya,
baru lanjut ke berikutnya. Menyalakan semuanya sekaligus lalu menemukan
regresi berarti Anda tidak tahu patch mana penyebabnya.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import HTTPException
from pydantic import BaseModel

DB_PATH = os.environ.get("ENPIDIX_DB", "surveillance.db")
SPOOL_MAX = 50_000

_srv: dict = {}          # globals() dari server.py
_stop = threading.Event()
_patch_state: dict[str, str] = {}
_started_at = time.time()


# ═══════════════════════════════════════════════════════════
# Konfigurasi fleet — disimpan di DB yang sama, satu baris
# ═══════════════════════════════════════════════════════════
_SCHEMA = """
CREATE TABLE IF NOT EXISTS fleet_config (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    engine_id       TEXT DEFAULT '',
    site_id         TEXT DEFAULT '',
    site_name       TEXT DEFAULT '',
    regional_url    TEXT DEFAULT '',
    regional_key    TEXT DEFAULT '',
    hw_model        TEXT DEFAULT '',
    enabled         INTEGER DEFAULT 0,
    patch_nvdec     INTEGER DEFAULT 0,
    patch_streamcopy INTEGER DEFAULT 0,
    patch_parallel_ai INTEGER DEFAULT 0,
    ai_slots        INTEGER DEFAULT 3,
    updated_at      TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS fleet_spool (
    event_id   TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    attempts   INTEGER DEFAULT 0,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fleet_spool ON fleet_spool(created_at);
"""

_db_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _db_lock:
        c = _conn()
        c.executescript(_SCHEMA)
        c.execute(
            "INSERT OR IGNORE INTO fleet_config(id, engine_id, site_id) VALUES (1, ?, ?)",
            (os.environ.get("ENPIDIX_ENGINE_ID", ""), os.environ.get("ENPIDIX_SITE_ID", "")),
        )
        c.commit()
        c.close()


def get_config() -> dict:
    with _db_lock:
        c = _conn()
        row = c.execute("SELECT * FROM fleet_config WHERE id=1").fetchone()
        c.close()
    return dict(row) if row else {}


def set_config(patch: dict) -> dict:
    allowed = {
        "engine_id", "site_id", "site_name", "regional_url", "regional_key",
        "hw_model", "enabled", "patch_nvdec", "patch_streamcopy",
        "patch_parallel_ai", "ai_slots",
    }
    fields = {k: v for k, v in patch.items() if k in allowed}
    if not fields:
        return get_config()
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    sets = ", ".join(f"{k}=?" for k in fields)
    with _db_lock:
        c = _conn()
        c.execute(f"UPDATE fleet_config SET {sets} WHERE id=1", list(fields.values()))
        c.commit()
        c.close()
    return get_config()


# ═══════════════════════════════════════════════════════════
# Store & Forward
# ═══════════════════════════════════════════════════════════
def emit(kind: str, camera_id: str = "-", label: str = "", confidence: float = 0.0, meta: dict | None = None) -> None:
    """Masukkan event ke antrean lokal. Aman dipanggil dari thread mana pun."""
    cfg = get_config()
    if not cfg.get("enabled"):
        return
    payload = {
        "event_id": str(uuid.uuid4()),
        "site_id": cfg.get("site_id") or "site-unset",
        "engine_id": cfg.get("engine_id") or "engine-unset",
        "camera_id": str(camera_id),
        "kind": kind,
        "label": (label or "")[:200],
        "confidence": round(float(confidence or 0), 3),
        "ts": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
    }
    try:
        with _db_lock:
            c = _conn()
            c.execute(
                "INSERT OR IGNORE INTO fleet_spool(event_id, created_at, payload) VALUES (?,?,?)",
                (payload["event_id"], time.time(), json.dumps(payload)),
            )
            n = c.execute("SELECT COUNT(*) FROM fleet_spool").fetchone()[0]
            if n > SPOOL_MAX:
                # Saat WAN putus berhari-hari, event terbaru lebih berharga
                # daripada event minggu lalu.
                c.execute(
                    "DELETE FROM fleet_spool WHERE event_id IN "
                    "(SELECT event_id FROM fleet_spool ORDER BY created_at ASC LIMIT ?)",
                    (n - SPOOL_MAX,),
                )
            c.commit()
            c.close()
    except sqlite3.Error as exc:
        print(f"[FLEET ⚠️] Gagal menyimpan event ke spool: {exc}")


def spool_pending() -> int:
    try:
        with _db_lock:
            c = _conn()
            n = c.execute("SELECT COUNT(*) FROM fleet_spool").fetchone()[0]
            c.close()
        return n
    except sqlite3.Error:
        return -1


_fwd_state = {"connected": False, "last_error": "", "last_success": 0.0, "sent_total": 0}


def _flush_once(cfg: dict) -> None:
    with _db_lock:
        c = _conn()
        rows = c.execute(
            "SELECT event_id, payload FROM fleet_spool ORDER BY created_at ASC LIMIT 50"
        ).fetchall()
        c.close()
    if not rows:
        return

    ids = [r["event_id"] for r in rows]
    batch = {
        "proto": "1.0",
        "engine_id": cfg["engine_id"],
        "site_id": cfg["site_id"],
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "events": [json.loads(r["payload"]) for r in rows],
    }
    try:
        r = requests.post(
            cfg["regional_url"].rstrip("/") + "/api/ingest/events",
            json=batch,
            headers={"X-API-Key": cfg.get("regional_key", "")},
            timeout=20,
        )
        if r.ok:
            with _db_lock:
                c = _conn()
                c.executemany("DELETE FROM fleet_spool WHERE event_id=?", [(i,) for i in ids])
                c.commit()
                c.close()
            _fwd_state.update(connected=True, last_error="", last_success=time.time())
            _fwd_state["sent_total"] += len(ids)
        else:
            _fwd_state.update(connected=False, last_error=f"HTTP {r.status_code}: {r.text[:150]}")
    except requests.RequestException as exc:
        _fwd_state.update(connected=False, last_error=str(exc)[:200])


def build_heartbeat() -> dict:
    """Gabungkan telemetri sistem yang sudah ada di server.py dengan data Jetson."""
    cfg = get_config()
    snap = {}
    collector = _srv.get("_collect_system_snapshot")
    if collector:
        try:
            snap = collector()
        except Exception:  # noqa: BLE001
            snap = {}

    cams = []
    workers = _srv.get("camera_workers") or {}
    try:
        for cid, w in list(workers.items()):
            perf = getattr(w, "perf", {}) or {}
            cams.append({
                "camera_id": str(cid),
                "name": getattr(w, "name", "") or str(cid),
                "online": bool(getattr(w, "running", False)),
                "ai_enabled": bool(perf.get("yolo_ms_avg")),
                "capture_fps": float(perf.get("fps", 0) or 0),
                "infer_fps": float(perf.get("yolo_fps", 0) or 0),
                "last_frame_age_sec": -1.0,
                "recording": False,
            })
    except Exception:  # noqa: BLE001
        pass

    tg = _tegrastats()
    return {
        "proto": "1.0",
        "engine_id": cfg.get("engine_id") or "engine-unset",
        "site_id": cfg.get("site_id") or "site-unset",
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": tg.get("model") or cfg.get("hw_model") or "",
        "jetpack": tg.get("jetpack", ""),
        "uptime_sec": round(time.time() - _started_at, 1),
        "cpu_pct": float((snap.get("cpu") or {}).get("total_pct", 0) or 0),
        "mem_pct": float((snap.get("memory") or {}).get("pct", 0) or 0),
        "disk_pct": float((snap.get("disk") or {}).get("pct", 0) or 0),
        "gpu_pct": tg.get("gpu_pct", 0.0),
        "temp_c": tg.get("temp_c", 0.0),
        "power_w": tg.get("power_w", 0.0),
        "engine_backend": _patch_state.get("backend", "pytorch"),
        "decoder": _patch_state.get("decoder", "cpu"),
        "spool_pending": max(0, spool_pending()),
        "cameras": cams,
    }


def _forwarder_loop() -> None:
    last_beat = 0.0
    while not _stop.is_set():
        cfg = get_config()
        if cfg.get("enabled") and cfg.get("regional_url") and cfg.get("engine_id"):
            _flush_once(cfg)
            if time.time() - last_beat >= 30:
                try:
                    requests.post(
                        cfg["regional_url"].rstrip("/") + "/api/ingest/heartbeat",
                        json=build_heartbeat(),
                        headers={"X-API-Key": cfg.get("regional_key", "")},
                        timeout=12,
                    )
                    last_beat = time.time()
                except requests.RequestException as exc:
                    _fwd_state["last_error"] = str(exc)[:200]
                    last_beat = time.time()
        _stop.wait(5)


# ═══════════════════════════════════════════════════════════
# Telemetri Jetson
# ═══════════════════════════════════════════════════════════
_tg_state = {"gpu_pct": 0.0, "temp_c": 0.0, "power_w": 0.0, "model": "", "jetpack": ""}


def _tegrastats() -> dict:
    return dict(_tg_state)


def _start_tegrastats() -> None:
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            _tg_state["model"] = Path(p).read_text(errors="ignore").strip("\x00").strip()
            break
        except OSError:
            continue
    try:
        _tg_state["jetpack"] = Path("/etc/nv_tegra_release").read_text(errors="ignore").splitlines()[0].strip()
    except OSError:
        pass

    if not shutil.which("tegrastats"):
        return

    def loop() -> None:
        try:
            proc = subprocess.Popen(["tegrastats", "--interval", "2000"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:  # type: ignore[union-attr]
                if m := re.search(r"GR3D_FREQ (\d+)%", line):
                    _tg_state["gpu_pct"] = float(m.group(1))
                temps = [float(t) for t in re.findall(r"(?:CPU|GPU|SOC\d?|tj)@([\d.]+)C", line)]
                if temps:
                    _tg_state["temp_c"] = max(temps)
                if m := re.search(r"(?:VDD_IN|POM_5V_IN|VDD_GPU_SOC) (\d+)mW", line):
                    _tg_state["power_w"] = round(float(m.group(1)) / 1000, 1)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=loop, name="tegrastats", daemon=True).start()


# ═══════════════════════════════════════════════════════════
# Patch performa — masing-masing terpisah dan bisa dibatalkan
# ═══════════════════════════════════════════════════════════
def _gst_pipeline(url: str, codec: str, w: int = 704, h: int = 576) -> str:
    depay = "rtph265depay ! h265parse" if codec == "h265" else "rtph264depay ! h264parse"
    return (f"rtspsrc location={url} latency=200 protocols=tcp drop-on-latency=true ! "
            f"{depay} ! nvv4l2decoder enable-max-performance=1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


def _apply_nvdec() -> str:
    cv2 = _srv.get("cv2")
    Worker = _srv.get("CameraWorker")
    if not cv2 or not Worker:
        return "gagal: CameraWorker/cv2 tidak ditemukan"
    if "GStreamer:                   YES" not in cv2.getBuildInformation().replace("  ", "  "):
        if "gstreamer" not in cv2.getBuildInformation().lower():
            return "gagal: OpenCV ini dibuild tanpa GStreamer — pakai python3-opencv bawaan JetPack"

    if not hasattr(Worker, "_open_capture_orig"):
        Worker._open_capture_orig = Worker._open_capture

    def _open_capture(self):
        for codec in ("h264", "h265"):
            try:
                cap = cv2.VideoCapture(_gst_pipeline(self.rtsp_url, codec), cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    _patch_state["decoder"] = "nvdec"
                    return cap
            except Exception:  # noqa: BLE001
                pass
        _patch_state["decoder"] = "cpu"
        return Worker._open_capture_orig(self)

    Worker._open_capture = _open_capture
    _patch_state["decoder"] = "nvdec"
    return "aktif"


def _revert_nvdec() -> str:
    Worker = _srv.get("CameraWorker")
    if Worker and hasattr(Worker, "_open_capture_orig"):
        Worker._open_capture = Worker._open_capture_orig
    _patch_state["decoder"] = "cpu"
    return "nonaktif"


_ENCODE_FLAGS = {"-c:v", "-preset", "-crf", "-pix_fmt", "-vf", "-b:v", "-maxrate", "-bufsize"}


def _to_stream_copy(cmd: list) -> list:
    """Buang seluruh argumen encoding, ganti dengan salinan paket apa adanya."""
    out, skip = [], False
    for i, a in enumerate(cmd):
        if skip:
            skip = False
            continue
        if a in _ENCODE_FLAGS:
            skip = True
            continue
        out.append(a)
    # Sisipkan -c copy tepat sebelum pola output (argumen terakhir)
    return out[:-1] + ["-c", "copy", out[-1]]


def _apply_stream_copy() -> str:
    Rec = _srv.get("CameraRecorder")
    if not Rec:
        return "gagal: CameraRecorder tidak ditemukan"
    if not hasattr(Rec, "_run_loop_orig"):
        Rec._run_loop_orig = Rec._run_loop

    def _run_loop(self, cmd: list):
        try:
            cmd = _to_stream_copy(list(cmd))
        except Exception as exc:  # noqa: BLE001
            print(f"[FLEET ⚠️] Gagal mengubah cmd ke stream copy, pakai asli: {exc}")
        return Rec._run_loop_orig(self, cmd)

    Rec._run_loop = _run_loop
    return "aktif — rekaman berikutnya memakai -c copy (restart rekaman untuk menerapkan)"


def _revert_stream_copy() -> str:
    Rec = _srv.get("CameraRecorder")
    if Rec and hasattr(Rec, "_run_loop_orig"):
        Rec._run_loop = Rec._run_loop_orig
    return "nonaktif"


class _Slots:
    """Pengganti yolo_lock: izinkan N inferensi bersamaan, bukan satu per satu."""

    def __init__(self, n: int):
        self.n = n
        self._sem = threading.Semaphore(n)

    def __enter__(self):
        self._sem.acquire()
        return self

    def __exit__(self, *a):
        self._sem.release()
        return False


def _apply_parallel_ai(slots: int) -> str:
    if "yolo_lock" not in _srv:
        return "gagal: yolo_lock tidak ditemukan"
    if "_yolo_lock_orig" not in _srv:
        _srv["_yolo_lock_orig"] = _srv["yolo_lock"]
    _srv["yolo_lock"] = _Slots(max(1, int(slots)))
    return f"aktif — {slots} inferensi paralel"


def _revert_parallel_ai() -> str:
    if "_yolo_lock_orig" in _srv:
        _srv["yolo_lock"] = _srv["_yolo_lock_orig"]
    return "nonaktif — inferensi kembali diserialkan"


def apply_patches(cfg: dict | None = None) -> dict:
    cfg = cfg or get_config()
    result = {
        "nvdec": _apply_nvdec() if cfg.get("patch_nvdec") else _revert_nvdec(),
        "stream_copy": _apply_stream_copy() if cfg.get("patch_streamcopy") else _revert_stream_copy(),
        "parallel_ai": _apply_parallel_ai(cfg.get("ai_slots", 3)) if cfg.get("patch_parallel_ai") else _revert_parallel_ai(),
    }
    _patch_state.update({k: v for k, v in result.items()})
    return result


# ═══════════════════════════════════════════════════════════
# Jembatan event: bungkus pengirim event NX yang sudah ada
# ═══════════════════════════════════════════════════════════
def _bridge_events() -> None:
    """
    Setiap event AI di server.py sudah melewati `kirim_event_ke_nx`.
    Membungkusnya berarti seluruh deteksi lama (api, merokok, wajah, counting,
    Dahua native) otomatis ikut mengalir ke regional — tanpa menyentuh
    satu pun baris logika deteksi.
    """
    orig = _srv.get("kirim_event_ke_nx")
    if not callable(orig) or getattr(orig, "_fleet_wrapped", False):
        return

    def wrapped(caption: str, description: str = "", nx_camera_id: str = ""):
        try:
            kind = "object_detected"
            low = (caption or "").lower()
            if "api" in low or "fire" in low or "asap" in low or "smoke" in low:
                kind = "fire_smoke"
            elif "masuk" in low or "keluar" in low or "lintas" in low:
                kind = "line_crossing"
            elif "plat" in low or "anpr" in low:
                kind = "anpr"
            emit(kind, camera_id=nx_camera_id or "-", label=caption,
                 confidence=0.0, meta={"description": (description or "")[:300]})
        except Exception:  # noqa: BLE001
            pass
        return orig(caption, description, nx_camera_id)

    wrapped._fleet_wrapped = True  # type: ignore[attr-defined]
    _srv["kirim_event_ke_nx"] = wrapped


# ═══════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════
class FleetConfigIn(BaseModel):
    engine_id: str | None = None
    site_id: str | None = None
    site_name: str | None = None
    regional_url: str | None = None
    regional_key: str | None = None
    hw_model: str | None = None
    enabled: int | None = None
    patch_nvdec: int | None = None
    patch_streamcopy: int | None = None
    patch_parallel_ai: int | None = None
    ai_slots: int | None = None


def install(app, server_globals: dict) -> None:
    global _srv
    _srv = server_globals

    _init_db()
    _start_tegrastats()
    _bridge_events()
    apply_patches()
    threading.Thread(target=_forwarder_loop, name="fleet-forwarder", daemon=True).start()

    cfg = get_config()
    print(f"[FLEET ✅] engine={cfg.get('engine_id') or '(belum diset)'} "
          f"regional={cfg.get('regional_url') or '(belum diset)'} "
          f"patch={_patch_state.get('decoder')}")

    # ── endpoint ──
    @app.get("/api/fleet/config")
    async def fleet_get_config():
        cfg = get_config()
        cfg["regional_key"] = "••••••" if cfg.get("regional_key") else ""
        return cfg

    @app.post("/api/fleet/config")
    async def fleet_set_config(body: FleetConfigIn):
        patch = {k: v for k, v in body.model_dump().items() if v is not None}
        # Field key dikosongkan dari form edit = jangan hapus yang tersimpan
        if patch.get("regional_key") in ("", "••••••"):
            patch.pop("regional_key", None)
        cfg = set_config(patch)
        results = apply_patches(cfg)
        cfg["regional_key"] = "••••••" if cfg.get("regional_key") else ""
        return {"status": "ok", "config": cfg, "patches": results}

    @app.get("/api/fleet/status")
    async def fleet_status():
        cfg = get_config()
        return {
            "enabled": bool(cfg.get("enabled")),
            "engine_id": cfg.get("engine_id", ""),
            "site_id": cfg.get("site_id", ""),
            "regional_url": cfg.get("regional_url", ""),
            "connected": _fwd_state["connected"],
            "last_error": _fwd_state["last_error"],
            "last_success": _fwd_state["last_success"],
            "sent_total": _fwd_state["sent_total"],
            "spool_pending": spool_pending(),
            "patches": _patch_state,
            "telemetry": _tegrastats(),
        }

    @app.get("/api/fleet/health")
    async def fleet_health():
        """Dipanggil regional server tiap 30 detik. Format sama dengan edge-jetson."""
        return build_heartbeat()

    # Alias supaya regional server bisa memproxy tanpa perlu tahu ini server lama
    @app.get("/api/health")
    async def fleet_health_alias():
        return build_heartbeat()

    @app.post("/api/fleet/test")
    async def fleet_test():
        cfg = get_config()
        if not cfg.get("regional_url"):
            raise HTTPException(400, "URL regional belum diisi")
        t0 = time.time()
        try:
            r = requests.post(
                cfg["regional_url"].rstrip("/") + "/api/ingest/heartbeat",
                json=build_heartbeat(),
                headers={"X-API-Key": cfg.get("regional_key", "")},
                timeout=12,
            )
            return {
                "reachable": r.ok,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "detail": "Terhubung" if r.ok else f"HTTP {r.status_code}: {r.text[:150]}",
            }
        except requests.RequestException as exc:
            return {"reachable": False, "latency_ms": 0, "detail": str(exc)[:200]}

    @app.post("/api/fleet/flush")
    async def fleet_flush():
        cfg = get_config()
        if not cfg.get("regional_url"):
            raise HTTPException(400, "URL regional belum diisi")
        _flush_once(cfg)
        return {"status": "ok", "spool_pending": spool_pending(), "connected": _fwd_state["connected"]}


def shutdown() -> None:
    _stop.set()
