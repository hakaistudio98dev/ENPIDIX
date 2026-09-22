"""
=============================================================================
ENPIDIX — Modul Parking Analytics (pembacaan nyata dari CCTV)
=============================================================================
Modul ini membaca okupansi slot parkir dan menghitung kendaraan dari deteksi
AI kamera yang sudah berjalan di server.py — bukan simulasi.

BAGAIMANA DATANYA NYATA
-----------------------
1. CameraWorker sudah menjalankan YOLO tiap siklus dan menyimpan hasilnya di
   `worker.latest_detections` (koordinat sudah dinormalisasi 0-1 terhadap
   frame) serta angka lintasan di `worker.counts`.
2. Modul ini punya satu thread pemantau (`ParkingWatcher`) yang membaca dua
   nilai itu tiap detik, TANPA menjalankan inferensi tambahan. Jadi tidak ada
   beban YOLO baru sama sekali — tetap satu inferensi per kamera.
3. Okupansi slot: titik jangkar kendaraan (tengah-bawah kotak deteksi) diuji
   apakah berada di dalam poligon slot yang digambar operator di atas gambar
   CCTV. Perubahan baru ditulis setelah bertahan PARKING_CONFIRM_FRAMES siklus
   (anti-flicker, pola yang sama dengan deteksi api/asap).
4. Perhitungan kendaraan gerbang: selisih `worker.counts` antar siklus untuk
   kamera yang ditetapkan sebagai kamera gerbang di denah — angkanya berasal
   dari virtual line counting yang sudah ada, jadi konsisten dengan dashboard
   utama.

SYARAT AGAR TERBACA
-------------------
- Kamera parkir harus punya toggle **Vehicle Counting** aktif (itulah yang
  membuat CameraWorker menghasilkan kotak kendaraan). Endpoint /health di
  bawah akan memberi tahu kamera mana yang belum aktif.
- Tiap slot harus punya poligon yang digambar di atas gambar kamera lewat tab
  "Pantauan CCTV" di dashboard parkir.

CARA PASANG (3 baris di server.py, setelah `camera_manager` dibuat):

    import parking_api
    parking_api.init(get_conn, nx_alert=lapor_deteksi_nx, camera_manager=camera_manager)
    app.include_router(parking_api.router)

Halaman dashboard-nya:

    @app.get("/parking", response_class=HTMLResponse)
    async def parking_page():
        path = os.path.join(os.path.dirname(__file__), "parking3d.html")
        return HTMLResponse(open(path, encoding="utf-8").read())

Tidak ada baris lain di server.py yang perlu diubah.
=============================================================================
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from math import atan2, degrees, hypot
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/parking", tags=["parking"])

# =============================================================================
# ⚙️ KONFIGURASI
# =============================================================================
# Berapa siklus pemantauan berturut-turut sebelum status slot berubah.
# Dengan PARKING_POLL_SEC=1.0, nilai 6 berarti perubahan harus bertahan 6 detik.
PARKING_CONFIRM_FRAMES = int(os.environ.get("PARKING_CONFIRM_FRAMES", "6"))
PARKING_POLL_SEC = float(os.environ.get("PARKING_POLL_SEC", "1.0"))

# Titik jangkar kendaraan di dalam kotak deteksi: 0 = tepi atas, 1 = tepi bawah.
# 0.75 mendekati titik roda menyentuh aspal, jauh lebih akurat daripada titik
# tengah kotak saat kamera dipasang tinggi dan mobil terlihat memanjang.
PARKING_ANCHOR_Y = float(os.environ.get("PARKING_ANCHOR_Y", "0.75"))

# Batas durasi parkir sebelum ditandai "lewat batas waktu" (menit). 0 = mati.
PARKING_OVERSTAY_MIN = int(os.environ.get("PARKING_OVERSTAY_MIN", "480"))

# Tipe kotak deteksi kendaraan yang dihasilkan CameraWorker (prefix "count_").
VEHICLE_BOX_TYPES = {"count_car", "count_motorcycle", "count_motorbike", "count_bus", "count_truck"}
VEHICLE_CLASSES = {"car", "motorcycle", "motorbike", "bus", "truck"}
SLOT_TYPES = ("car", "moto", "disabled", "reserved")
STATUSES = ("free", "occupied", "overstay", "illegal", "reserved")

_get_conn: Optional[Callable] = None
_nx_alert: Optional[Callable] = None
_cam_mgr = None
_lock = threading.Lock()


# =============================================================================
# 🗄️ DATABASE
# =============================================================================
def init(get_conn: Callable, nx_alert: Optional[Callable] = None, camera_manager=None,
         start_watcher: bool = True):
    """Sambungkan modul ke SQLite, pelapor NX, dan daftar kamera aktif server."""
    global _get_conn, _nx_alert, _cam_mgr
    _get_conn = get_conn
    _nx_alert = nx_alert
    _cam_mgr = camera_manager
    _init_schema()
    watcher.reload()
    if start_watcher and camera_manager is not None:
        watcher.start()


def _conn():
    if _get_conn is None:
        raise RuntimeError("parking_api.init(get_conn) belum dipanggil.")
    return _get_conn()


def _quarantine_incompatible(conn):
    """Modul ini memakai id slot berupa teks ("A-01"). Kalau di database sudah
    ada tabel parking_slots dari implementasi parkir lama dengan id INTEGER,
    tabel itu tidak bisa dipakai maupun ditambal — jadi kita ganti namanya
    supaya datanya tetap aman, lalu tabel baru dibuat di sebelahnya."""
    info = conn.execute("PRAGMA table_info(parking_slots)").fetchall()
    if not info:
        return
    cols = {r[1]: (r[2] or "").upper() for r in info}
    if "id" in cols and "INT" not in cols["id"]:
        return                                   # sudah cocok, cukup ditambal kolomnya
    backup = "parking_slots_lama_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    conn.execute(f"ALTER TABLE parking_slots RENAME TO {backup}")
    conn.commit()
    print(f"[Parking] Tabel parking_slots lama tidak kompatibel (id bertipe "
          f"{cols.get('id', 'tidak ada')}). Data lama disimpan sebagai '{backup}', "
          f"tabel baru dibuat. Denah perlu disusun ulang lewat menu Atur denah.")


def _init_schema():
    conn = _conn()
    _quarantine_incompatible(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parking_areas (
            name       TEXT PRIMARY KEY,
            width_m    REAL NOT NULL DEFAULT 40,
            length_m   REAL NOT NULL DEFAULT 30,
            grid_m     REAL NOT NULL DEFAULT 5,
            gates      TEXT,
            sort_order INTEGER DEFAULT 0
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parking_slots (
            id         TEXT PRIMARY KEY,
            level      TEXT NOT NULL DEFAULT 'P1',
            zone       TEXT NOT NULL DEFAULT 'A',
            type       TEXT NOT NULL DEFAULT 'car',
            mode       TEXT NOT NULL DEFAULT 'perp',  -- perp | parallel | angled
            x          REAL NOT NULL DEFAULT 0,       -- meter, dari sudut kiri-atas area
            y          REAL NOT NULL DEFAULT 0,
            w          REAL NOT NULL DEFAULT 2.4,     -- lebar slot, meter
            l          REAL NOT NULL DEFAULT 5.0,     -- panjang slot, meter
            rot        REAL NOT NULL DEFAULT 0,       -- derajat, 0 = menghadap atas denah
            camera_id  INTEGER,
            polygon    TEXT,
            status     TEXT NOT NULL DEFAULT 'free',
            plate      TEXT,
            since      TEXT,
            updated_at TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parking_cameras (
            camera_id INTEGER NOT NULL,
            level     TEXT NOT NULL DEFAULT 'P1',
            x REAL DEFAULT 0, y REAL DEFAULT 0, rot REAL DEFAULT 0,
            fov REAL DEFAULT 90, range_m REAL DEFAULT 25, height_m REAL DEFAULT 4,
            PRIMARY KEY (camera_id, level)
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parking_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            at            TEXT NOT NULL,
            direction     TEXT NOT NULL,
            source        TEXT NOT NULL DEFAULT 'slot',   -- slot | gate | manual
            camera_id     INTEGER,
            camera_name   TEXT,
            gate          TEXT,
            slot_id       TEXT,
            plate         TEXT,
            vehicle_class TEXT
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pevt_at ON parking_events(at)")

    # -------------------------------------------------------------------------
    # Migrasi. Tabel parkir mungkin sudah ada dari implementasi parkir lama
    # dengan susunan kolom berbeda, sehingga CREATE TABLE IF NOT EXISTS di atas
    # tidak berpengaruh apa-apa. Di sini setiap kolom yang kurang ditambahkan.
    # -------------------------------------------------------------------------
    _ensure_columns(conn, "parking_slots", {
        "level": "TEXT NOT NULL DEFAULT 'P1'", "zone": "TEXT NOT NULL DEFAULT 'A'",
        "type": "TEXT NOT NULL DEFAULT 'car'", "mode": "TEXT NOT NULL DEFAULT 'perp'",
        "x": "REAL NOT NULL DEFAULT 0", "y": "REAL NOT NULL DEFAULT 0",
        "w": "REAL NOT NULL DEFAULT 2.4", "l": "REAL NOT NULL DEFAULT 5.0",
        "rot": "REAL NOT NULL DEFAULT 0", "camera_id": "INTEGER",
        "polygon": "TEXT", "status": "TEXT NOT NULL DEFAULT 'free'",
        "plate": "TEXT", "since": "TEXT", "updated_at": "TEXT",
    })
    _ensure_columns(conn, "parking_areas", {
        "width_m": "REAL NOT NULL DEFAULT 40", "length_m": "REAL NOT NULL DEFAULT 30",
        "grid_m": "REAL NOT NULL DEFAULT 5", "gates": "TEXT", "sort_order": "INTEGER DEFAULT 0",
    })
    _ensure_columns(conn, "parking_cameras", {
        "level": "TEXT NOT NULL DEFAULT 'P1'", "x": "REAL DEFAULT 0", "y": "REAL DEFAULT 0",
        "rot": "REAL DEFAULT 0", "fov": "REAL DEFAULT 90",
        "range_m": "REAL DEFAULT 25", "height_m": "REAL DEFAULT 4",
    })
    _ensure_columns(conn, "parking_events", {
        "at": "TEXT", "direction": "TEXT", "source": "TEXT NOT NULL DEFAULT 'slot'",
        "camera_id": "INTEGER", "camera_name": "TEXT", "gate": "TEXT",
        "slot_id": "TEXT", "plate": "TEXT", "vehicle_class": "TEXT",
    })

    if not conn.execute("SELECT 1 FROM parking_areas LIMIT 1").fetchone():
        conn.execute("INSERT INTO parking_areas (name,width_m,length_m,grid_m,gates,sort_order) VALUES (?,?,?,?,?,0)",
                     ("P1", 40, 30, 5, json.dumps({"in": {"x": 3, "y": 1, "rot": 0, "camera_id": None},
                                                   "out": {"x": 33, "y": 1, "rot": 0, "camera_id": None}})))
    conn.commit()
    conn.close()


def _ensure_columns(conn, table: str, columns: dict):
    """Tambahkan kolom yang belum ada pada tabel lama. SQLite hanya mengizinkan
    ALTER TABLE ADD COLUMN dengan nilai bawaan konstan, dan itu cukup di sini."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    have = {r[1] for r in info}
    for col, ddl in columns.items():
        if col not in have:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {ddl}')
            print(f"[Parking] kolom '{col}' ditambahkan ke {table}")


# =============================================================================
# 📦 MODEL
# =============================================================================
class AreaIn(BaseModel):
    name: str
    width_m: float = 40
    length_m: float = 30
    grid_m: float = 5
    gates: Optional[dict] = None


class SlotIn(BaseModel):
    id: str
    level: str = "P1"
    zone: str = "A"
    type: str = "car"
    mode: str = "perp"                  # perp | parallel | angled
    x: float = 0
    y: float = 0
    w: Optional[float] = None           # kosong = pakai ukuran baku jenis + pola
    l: Optional[float] = None
    rot: float = 0
    camera_id: Optional[int] = None
    polygon: Optional[list] = None


class CamIn(BaseModel):
    camera_id: int
    level: str = "P1"
    x: float = 0
    y: float = 0
    rot: float = 0
    fov: float = 90
    range: float = 25
    height: float = 4


class LayoutIn(BaseModel):
    areas: list
    slots: list = []
    cameras: list = []


class RoiIn(BaseModel):
    """Poligon slot di atas gambar kamera, koordinat 0-1 relatif frame."""
    slot_id: str
    polygon: Optional[list] = None      # None / [] = hapus poligon


class RoiBulk(BaseModel):
    rois: list


class SlotStatusIn(BaseModel):
    status: str
    plate: Optional[str] = None


class EventIn(BaseModel):
    direction: str
    camera_id: Optional[int] = None
    camera_name: Optional[str] = None
    gate: Optional[str] = None
    slot_id: Optional[str] = None
    plate: Optional[str] = None
    vehicle_class: Optional[str] = "car"


# =============================================================================
# 📐 GEOMETRI
# =============================================================================
# Ukuran baku slot (lebar, panjang) dalam meter, per jenis kendaraan dan pola parkir.
# Slot paralel lebih panjang karena mobil butuh ruang manuver maju-mundur.
BASE_SIZE = {
    "car":      {"perp": (2.4, 5.0), "parallel": (2.4, 6.0), "angled": (2.4, 5.0)},
    "moto":     {"perp": (1.1, 2.2), "parallel": (1.1, 2.6), "angled": (1.1, 2.2)},
    "disabled": {"perp": (3.6, 5.0), "parallel": (3.6, 6.5), "angled": (3.6, 5.0)},
    "reserved": {"perp": (2.4, 5.0), "parallel": (2.4, 6.0), "angled": (2.4, 5.0)},
}
SLOT_MODES = ("perp", "parallel", "angled")


def slot_size(slot: dict) -> tuple:
    """Ukuran slot: pakai yang tersimpan kalau ada, kalau tidak ambil ukuran baku."""
    base = BASE_SIZE.get(slot.get("type", "car"), BASE_SIZE["car"])
    bw, bl = base.get(slot.get("mode") or "perp", base["perp"])
    return (slot.get("w") or bw, slot.get("l") or bl)


def _angle_to(cx, cy, x, y):
    return (degrees(atan2(x - cx, -(y - cy))) + 360) % 360


def _angle_diff(a, b):
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d


def covers(cam: dict, slot: dict) -> bool:
    """Slot terpantau bila titik tengahnya masuk kerucut FOV kamera di denah."""
    w, l = slot_size(slot)
    cx, cy = slot["x"] + w / 2, slot["y"] + l / 2
    dist = hypot(cx - cam["x"], cy - cam["y"])
    if dist > cam["range_m"]:
        return False
    if dist < 0.4:
        return True
    return _angle_diff(_angle_to(cam["x"], cam["y"], cx, cy), cam["rot"]) <= cam["fov"] / 2


def assign_coverage(slots: list, cams: list):
    """Petakan slot ke kamera. Poligon yang sudah digambar menang; kalau belum
    ada, hasil pemetaan denah dipakai sebagai usulan."""
    for s in slots:
        best, best_d = None, float("inf")
        for c in cams:
            if c["level"] != s["level"] or not covers(c, s):
                continue
            w, l = slot_size(s)
            d = hypot(s["x"] + w / 2 - c["x"], s["y"] + l / 2 - c["y"])
            if d < best_d:
                best, best_d = c, d
        s["covered"] = best is not None
        if best and not s.get("polygon"):
            s["camera_id"] = best["camera_id"]


def point_in_polygon(px: float, py: float, poly: list) -> bool:
    """Ray-casting — sama dengan counting polygon di server.py."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    x1, y1 = poly[0]
    for i in range(1, n + 1):
        x2, y2 = poly[i % n]
        if py > min(y1, y2) and py <= max(y1, y2) and px <= max(x1, x2) and y1 != y2:
            xin = (py - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or px <= xin:
                inside = not inside
        x1, y1 = x2, y2
    return inside


# =============================================================================
# 🌐 DENAH
# =============================================================================
def _read_layout():
    conn = _conn()
    areas = [{"name": r[0], "width_m": r[1], "length_m": r[2], "grid_m": r[3],
              "gates": json.loads(r[4]) if r[4] else {}}
             for r in conn.execute("SELECT name,width_m,length_m,grid_m,gates FROM parking_areas "
                                   "ORDER BY sort_order, name")]
    slots = [{"id": r[0], "level": r[1], "zone": r[2], "type": r[3], "mode": r[4],
              "x": r[5], "y": r[6], "w": r[7], "l": r[8], "rot": r[9],
              "camera_id": r[10], "polygon": json.loads(r[11]) if r[11] else None,
              "status": r[12], "plate": r[13], "since": r[14]}
             for r in conn.execute("SELECT id,level,zone,type,mode,x,y,w,l,rot,camera_id,polygon,"
                                   "status,plate,since FROM parking_slots ORDER BY level, zone, id")]
    cams = [{"camera_id": r[0], "level": r[1], "x": r[2], "y": r[3],
             "rot": r[4], "fov": r[5], "range_m": r[6], "height_m": r[7]}
            for r in conn.execute("SELECT camera_id,level,x,y,rot,fov,range_m,height_m FROM parking_cameras")]
    conn.close()
    assign_coverage(slots, cams)
    return areas, slots, cams


@router.get("/layout")
async def get_layout():
    areas, slots, cams = _read_layout()
    return {"areas": areas, "slots": slots,
            "cameras": [{"camera_id": c["camera_id"], "level": c["level"], "x": c["x"], "y": c["y"],
                         "rot": c["rot"], "fov": c["fov"], "range": c["range_m"], "height": c["height_m"]}
                        for c in cams],
            "blind_slots": [s["id"] for s in slots if not s["covered"]]}


@router.post("/layout")
async def save_layout(body: LayoutIn):
    """Simpan denah. Poligon ROI dan status okupansi slot lama dipertahankan —
    yang ditimpa hanya posisi, ukuran area, dan penempatan kamera."""
    areas = [AreaIn(**a) if isinstance(a, dict) else a for a in body.areas]
    slots = [SlotIn(**s) if isinstance(s, dict) else s for s in body.slots]
    cams = [CamIn(**c) if isinstance(c, dict) else c for c in body.cameras]

    if not areas:
        raise HTTPException(422, "Minimal satu area harus ada.")
    names = {a.name for a in areas}
    for s in slots:
        if s.level not in names:
            raise HTTPException(422, f"Slot {s.id} merujuk area '{s.level}' yang tidak ada.")
        if s.type not in SLOT_TYPES:
            raise HTTPException(422, f"Slot {s.id}: jenis '{s.type}' tidak dikenal.")
        if s.mode not in SLOT_MODES:
            raise HTTPException(422, f"Slot {s.id}: pola parkir '{s.mode}' tidak dikenal.")
    for c in cams:
        if c.level not in names:
            raise HTTPException(422, f"Kamera {c.camera_id} merujuk area '{c.level}' yang tidak ada.")
        if not (5 <= c.fov <= 360) or c.range <= 0:
            raise HTTPException(422, f"Kamera {c.camera_id}: sudut atau jangkauan tidak masuk akal.")

    now = datetime.now().isoformat()
    conn = _conn()
    conn.execute("DELETE FROM parking_areas")
    for i, a in enumerate(areas):
        conn.execute("INSERT INTO parking_areas (name,width_m,length_m,grid_m,gates,sort_order) VALUES (?,?,?,?,?,?)",
                     (a.name, a.width_m, a.length_m, a.grid_m, json.dumps(a.gates or {}), i))

    keep = {s.id for s in slots}
    for (sid,) in conn.execute("SELECT id FROM parking_slots").fetchall():
        if sid not in keep:
            conn.execute("DELETE FROM parking_slots WHERE id=?", (sid,))
    for s in slots:
        w, l = slot_size({"type": s.type, "mode": s.mode, "w": s.w, "l": s.l})
        conn.execute("""
            INSERT INTO parking_slots (id,level,zone,type,mode,x,y,w,l,rot,camera_id,status,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?, COALESCE((SELECT status FROM parking_slots WHERE id=?), 'free'), ?)
            ON CONFLICT(id) DO UPDATE SET
                level=excluded.level, zone=excluded.zone, type=excluded.type, mode=excluded.mode,
                x=excluded.x, y=excluded.y, w=excluded.w, l=excluded.l, rot=excluded.rot,
                camera_id=excluded.camera_id, updated_at=excluded.updated_at
        """, (s.id, s.level, s.zone, s.type, s.mode, s.x, s.y, w, l, s.rot, s.camera_id, s.id, now))
    conn.execute("DELETE FROM parking_cameras")
    for c in cams:
        conn.execute("INSERT INTO parking_cameras (camera_id,level,x,y,rot,fov,range_m,height_m) "
                     "VALUES (?,?,?,?,?,?,?,?)",
                     (c.camera_id, c.level, c.x, c.y, c.rot, c.fov, c.range, c.height))
    conn.commit()
    conn.close()
    watcher.reload()

    _, all_slots, _ = _read_layout()
    return {"status": "ok", "areas": len(areas), "slots": len(slots), "cameras": len(cams),
            "blind_slots": [s["id"] for s in all_slots if not s["covered"]]}


# =============================================================================
# 🎥 ROI — poligon slot di atas gambar CCTV
# =============================================================================
@router.get("/cameras/{cam_id}/rois")
async def get_rois(cam_id: int):
    """Slot yang ditugaskan ke kamera ini, beserta poligonnya di frame kamera."""
    conn = _conn()
    rows = conn.execute("SELECT id,level,zone,type,polygon,status,since FROM parking_slots "
                        "WHERE camera_id=? ORDER BY id", (cam_id,)).fetchall()
    conn.close()
    slots = [{"id": r[0], "level": r[1], "zone": r[2], "type": r[3],
              "polygon": json.loads(r[4]) if r[4] else None, "status": r[5], "since": r[6]} for r in rows]
    return {"camera_id": cam_id, "slots": slots,
            "drawn": sum(1 for s in slots if s["polygon"]),
            "pending": [s["id"] for s in slots if not s["polygon"]]}


@router.post("/cameras/{cam_id}/rois")
async def save_rois(cam_id: int, body: RoiBulk):
    """Simpan poligon slot hasil menggambar di atas gambar kamera. Begitu
    tersimpan, slot itu langsung dibaca pemantau dari deteksi kamera."""
    rois = [RoiIn(**r) if isinstance(r, dict) else r for r in body.rois]
    conn = _conn()
    known = {r[0] for r in conn.execute("SELECT id FROM parking_slots").fetchall()}
    try:
        for roi in rois:
            if roi.slot_id not in known:
                raise HTTPException(404, f"Slot {roi.slot_id} tidak terdaftar di denah.")
            if roi.polygon:
                if not (3 <= len(roi.polygon) <= 16):
                    raise HTTPException(422, f"Slot {roi.slot_id}: poligon butuh 3-16 titik.")
                for pt in roi.polygon:
                    if len(pt) != 2 or not (0 <= pt[0] <= 1 and 0 <= pt[1] <= 1):
                        raise HTTPException(422, f"Slot {roi.slot_id}: koordinat harus 0-1.")
            conn.execute("UPDATE parking_slots SET polygon=?, camera_id=?, updated_at=? WHERE id=?",
                         (json.dumps(roi.polygon) if roi.polygon else None, cam_id,
                          datetime.now().isoformat(), roi.slot_id))
        conn.commit()
    finally:
        conn.close()
    watcher.reload()
    return {"status": "ok", "saved": len(rois)}


@router.get("/cameras/{cam_id}/live")
async def camera_live(cam_id: int):
    """Kotak kendaraan terbaru dari kamera ini + status tiap ROI — dipakai editor
    untuk membuktikan slot benar-benar terbaca oleh kamera."""
    info = _vehicle_anchors(cam_id, with_boxes=True)
    conn = _conn()
    rows = conn.execute("SELECT id,polygon,status FROM parking_slots "
                        "WHERE camera_id=? AND polygon IS NOT NULL", (cam_id,)).fetchall()
    conn.close()
    live = []
    for sid, poly, status in rows:
        p = json.loads(poly)
        live.append({"slot_id": sid, "status": status,
                     "detected_now": any(point_in_polygon(a[0], a[1], p) for a in info["anchors"])})
    return {"camera_id": cam_id, "boxes": info["boxes"], "anchors": info["anchors"], "slots": live,
            "vehicle_count": info["count"], "worker_active": info["worker_active"],
            "vehicle_counting": info["vehicle_counting"], "last_seen": info["last_seen"]}


@router.get("/health")
async def health():
    """Diagnosa: kamera mana yang benar-benar sudah memasok data ke sistem parkir."""
    conn = _conn()
    rows = conn.execute("SELECT camera_id, COUNT(*), SUM(CASE WHEN polygon IS NOT NULL THEN 1 ELSE 0 END) "
                        "FROM parking_slots WHERE camera_id IS NOT NULL GROUP BY camera_id").fetchall()
    gates: dict = {}
    for (g,) in conn.execute("SELECT gates FROM parking_areas").fetchall():
        for k, v in (json.loads(g) if g else {}).items():
            if v and v.get("camera_id") is not None:
                gates.setdefault(int(v["camera_id"]), []).append(k)
    conn.close()

    out = []
    for cam_id, total, drawn in rows:
        drawn = drawn or 0
        info = _vehicle_anchors(int(cam_id))
        problems = []
        if not info["worker_active"]:
            problems.append("Kamera tidak aktif di server.")
        elif not info["vehicle_counting"]:
            problems.append("Vehicle Counting mati — nyalakan agar kendaraan terdeteksi.")
        if drawn == 0:
            problems.append("Belum ada slot yang digambar di gambar kamera.")
        elif drawn < total:
            problems.append(f"{total - drawn} slot belum digambar.")
        out.append({"camera_id": int(cam_id), "name": info["name"],
                    "worker_active": info["worker_active"], "vehicle_counting": info["vehicle_counting"],
                    "slots_assigned": total, "slots_drawn": drawn, "vehicles_now": info["count"],
                    "is_gate": gates.get(int(cam_id), []), "last_seen": info["last_seen"],
                    "problems": problems})
    for cam_id, kinds in gates.items():
        if any(o["camera_id"] == cam_id for o in out):
            continue
        info = _vehicle_anchors(cam_id)
        out.append({"camera_id": cam_id, "name": info["name"], "worker_active": info["worker_active"],
                    "vehicle_counting": info["vehicle_counting"], "slots_assigned": 0, "slots_drawn": 0,
                    "vehicles_now": info["count"], "is_gate": kinds, "last_seen": info["last_seen"],
                    "problems": ([] if info["vehicle_counting"]
                                 else ["Vehicle Counting mati — angka gerbang tidak akan bertambah."])})
    return {"cameras": out, "ready": bool(out) and all(not o["problems"] for o in out),
            "watcher_running": watcher.running, "last_cycle": watcher.last_cycle,
            "cycle_ms": watcher.cycle_ms, "poll_sec": PARKING_POLL_SEC,
            "confirm_cycles": PARKING_CONFIRM_FRAMES}


# =============================================================================
# 🌐 STATUS REAL-TIME
# =============================================================================
@router.get("/overview")
async def overview():
    areas, slots, cams = _read_layout()
    _apply_overstay(slots)
    busy = sum(1 for s in slots if s["status"] in ("occupied", "overstay"))
    return {"slots": slots, "areas": areas, "levels": [a["name"] for a in areas],
            "zones": sorted({s["zone"] for s in slots}),
            "summary": {"capacity": len(slots), "occupied": busy,
                        "free": sum(1 for s in slots if s["status"] == "free"),
                        "illegal": sum(1 for s in slots if s["status"] == "illegal"),
                        "blind": sum(1 for s in slots if not s["covered"]),
                        "unread": sum(1 for s in slots if not s["polygon"]),
                        "occupancy_pct": round(busy / len(slots) * 100) if slots else 0},
            "server_time": datetime.now().isoformat()}


@router.post("/slots/{slot_id}/status")
async def set_slot_status(slot_id: str, body: SlotStatusIn):
    if body.status not in STATUSES:
        raise HTTPException(422, "Status tidak dikenal.")
    _write_status(slot_id, body.status, body.plate)
    with _lock:
        watcher._current[slot_id] = body.status
    return {"status": "ok"}


@router.get("/flow")
async def flow(hours: int = 24):
    """Arus per jam. Memakai catatan gerbang bila ada (paling akurat); kalau
    belum ada kamera gerbang, memakai catatan masuk/keluar slot."""
    hours = max(1, min(48, hours))
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    conn = _conn()
    rows = conn.execute("SELECT at, direction, source FROM parking_events WHERE at >= ?", (since,)).fetchall()
    conn.close()
    use = "gate" if any(r[2] == "gate" for r in rows) else "slot"
    buckets: dict = {}
    for at, direction, source in rows:
        if source not in (use, "manual"):
            continue
        try:
            h = datetime.fromisoformat(at).hour
        except ValueError:
            continue
        b = buckets.setdefault(h, {"hour": h, "in": 0, "out": 0})
        b["in" if direction == "in" else "out"] += 1
    return {"source": use,
            "hours": [buckets.get(h, {"hour": h, "in": 0, "out": 0})
                      for h in range(0, datetime.now().hour + 1)]}


@router.get("/events")
async def list_events(limit: int = 20):
    limit = max(1, min(200, limit))
    conn = _conn()
    rows = conn.execute("SELECT at,direction,camera_id,camera_name,gate,slot_id,plate,vehicle_class,source "
                        "FROM parking_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"events": [{"at": r[0], "direction": r[1], "camera_id": r[2], "camera_name": r[3], "gate": r[4],
                        "slot_id": r[5], "plate": r[6], "vehicle_class": r[7], "source": r[8]} for r in rows]}


@router.post("/events")
async def add_event(body: EventIn):
    """Catat kendaraan masuk/keluar dari sumber lain — mis. event ANPR
    TrafficJunction Dahua yang membawa plat nomor."""
    if body.direction not in ("in", "out"):
        raise HTTPException(422, "direction harus 'in' atau 'out'.")
    data = body.dict()
    data.pop("direction", None)
    record_event(body.direction, source="gate", **data)
    return {"status": "ok"}


# =============================================================================
# ✍️ PENULIS DATA
# =============================================================================
def record_event(direction: str, source: str = "slot", camera_id=None, camera_name=None,
                 gate=None, slot_id=None, plate=None, vehicle_class="car"):
    conn = _conn()
    conn.execute("INSERT INTO parking_events "
                 "(at,direction,source,camera_id,camera_name,gate,slot_id,plate,vehicle_class) "
                 "VALUES (?,?,?,?,?,?,?,?,?)",
                 (datetime.now().isoformat(), direction, source, camera_id, camera_name,
                  gate, slot_id, plate, vehicle_class))
    conn.commit()
    conn.close()


def _write_status(slot_id: str, status: str, plate: Optional[str] = None):
    now = datetime.now().isoformat()
    conn = _conn()
    if status == "free":
        conn.execute("UPDATE parking_slots SET status=?, plate=NULL, since=NULL, updated_at=? WHERE id=?",
                     (status, now, slot_id))
    else:
        conn.execute("UPDATE parking_slots SET status=?, plate=COALESCE(?,plate), since=COALESCE(since,?), "
                     "updated_at=? WHERE id=?", (status, plate, now, now, slot_id))
    conn.commit()
    conn.close()


def _apply_overstay(slots: list):
    """Hitung saat dibaca — tidak perlu thread pemindai tambahan."""
    if PARKING_OVERSTAY_MIN <= 0:
        return
    limit = timedelta(minutes=PARKING_OVERSTAY_MIN)
    now = datetime.now()
    for s in slots:
        if s["status"] == "occupied" and s["since"]:
            try:
                if now - datetime.fromisoformat(s["since"]) > limit:
                    s["status"] = "overstay"
            except ValueError:
                pass


# =============================================================================
# 📡 MEMBACA DETEKSI DARI CameraWorker
# =============================================================================
def _vehicle_anchors(cam_id: int, with_boxes: bool = False) -> dict:
    """
    Ambil kotak kendaraan terbaru milik satu kamera dari CameraWorker.
    Tidak menjalankan inferensi apa pun — hanya membaca hasil yang sudah ada,
    jadi tidak menambah beban YOLO sama sekali.
    """
    empty = {"anchors": [], "boxes": [], "count": 0, "worker_active": False,
             "vehicle_counting": False, "name": f"Kamera {cam_id}", "last_seen": None}
    if _cam_mgr is None:
        return empty
    worker = _cam_mgr.get(cam_id)
    if worker is None:
        return empty

    with worker.lock:
        boxes = [b for b in worker.latest_detections if b.get("type") in VEHICLE_BOX_TYPES]
    frame_at = getattr(worker, "perf", {}).get("last_frame_at")

    anchors = [(round((b["x1"] + b["x2"]) / 2, 4),
                round(b["y1"] + (b["y2"] - b["y1"]) * PARKING_ANCHOR_Y, 4)) for b in boxes]
    return {"anchors": anchors, "boxes": boxes if with_boxes else [], "count": len(anchors),
            "worker_active": True,
            "vehicle_counting": bool(worker.ai_settings.get("vehicle_counting")),
            "name": worker.name,
            "last_seen": datetime.fromtimestamp(frame_at).isoformat() if frame_at else None}


def _vehicle_totals(worker) -> tuple:
    """Total lintasan kendaraan (masuk, keluar) dari counting line kamera ini."""
    with worker.lock:
        counts = dict(worker.counts)
    ci = sum(v.get("in", 0) for k, v in counts.items() if k in VEHICLE_CLASSES)
    co = sum(v.get("out", 0) for k, v in counts.items() if k in VEHICLE_CLASSES)
    return ci, co


# =============================================================================
# 🔁 PEMANTAU (satu thread untuk seluruh kamera parkir)
# =============================================================================
class ParkingWatcher:
    """
    Membaca deteksi kamera tiap PARKING_POLL_SEC detik dan menerjemahkannya
    jadi status slot + catatan masuk/keluar.

    Anti-flicker: perubahan status baru ditulis setelah bertahan
    PARKING_CONFIRM_FRAMES siklus — inilah yang membedakan pembacaan stabil dari
    angka yang berkedip tiap kali ada orang lewat di depan mobil.

    Hemat: satu thread untuk semua kamera, tanpa inferensi tambahan, dan DB
    hanya ditulis saat status benar-benar berubah — bukan tiap siklus.
    """

    def __init__(self):
        self._by_cam: dict = {}       # cam_id -> [{id, polygon}]
        self._pending: dict = {}      # slot_id -> (kandidat, hitungan)
        self._current: dict = {}      # slot_id -> status tertulis
        self._gate_cams: dict = {}    # cam_id -> ["in"] / ["out"]
        self._gate_last: dict = {}    # cam_id -> (in, out) snapshot terakhir
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.running = False
        self.last_cycle: Optional[str] = None
        self.cycle_ms = 0.0

    def reload(self):
        if _get_conn is None:
            return
        by_cam: dict = {}
        cur: dict = {}
        conn = _conn()
        for sid, cam_id, poly, status in conn.execute(
                "SELECT id,camera_id,polygon,status FROM parking_slots WHERE polygon IS NOT NULL"):
            if cam_id is None:
                continue
            try:
                pts = json.loads(poly)
            except Exception:
                continue
            if len(pts) < 3:
                continue
            by_cam.setdefault(int(cam_id), []).append({"id": sid, "polygon": pts})
            cur[sid] = status
        gate_cams: dict = {}
        for (g,) in conn.execute("SELECT gates FROM parking_areas").fetchall():
            for kind, v in (json.loads(g) if g else {}).items():
                if v and v.get("camera_id") is not None:
                    gate_cams.setdefault(int(v["camera_id"]), set()).add(kind)
        conn.close()
        with _lock:
            self._by_cam = by_cam
            self._current.update(cur)
            self._gate_cams = {k: sorted(v) for k, v in gate_cams.items()}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="parking-watcher")
        self._thread.start()
        self.running = True
        print(f"[Parking ✅] Pemantau berjalan — baca deteksi tiap {PARKING_POLL_SEC}s, "
              f"konfirmasi {PARKING_CONFIRM_FRAMES} siklus.")

    def stop(self):
        self._stop.set()
        self.running = False

    def _loop(self):
        time.sleep(3)   # beri waktu kamera membuka stream
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.cycle()
            except Exception as e:
                print(f"[Parking Watcher ERR] {e}")
            self.cycle_ms = round((time.time() - t0) * 1000, 1)
            self.last_cycle = datetime.now().isoformat()
            self._stop.wait(PARKING_POLL_SEC)

    def cycle(self):
        """Satu siklus pembacaan. Dipisah supaya bisa dipanggil langsung saat uji."""
        if _cam_mgr is None:
            return
        with _lock:
            by_cam = dict(self._by_cam)
            gate_cams = dict(self._gate_cams)

        # --- okupansi slot dari kotak kendaraan kamera ---
        for cam_id, slots in by_cam.items():
            info = _vehicle_anchors(cam_id)
            if not info["worker_active"] or not info["vehicle_counting"]:
                continue        # tidak ada pasokan deteksi; jangan menebak
            for slot in slots:
                hit = any(point_in_polygon(ax, ay, slot["polygon"]) for ax, ay in info["anchors"])
                self._vote(slot["id"], "occupied" if hit else "free", cam_id, info["name"])

        # --- perhitungan kendaraan gerbang dari counting line ---
        for cam_id, kinds in gate_cams.items():
            worker = _cam_mgr.get(cam_id)
            if worker is None or not worker.ai_settings.get("vehicle_counting"):
                continue
            ci, co = _vehicle_totals(worker)
            prev = self._gate_last.get(cam_id)
            self._gate_last[cam_id] = (ci, co)
            if prev is None:
                continue                    # siklus pertama hanya mengambil patokan
            d_in, d_out = ci - prev[0], co - prev[1]
            if d_in < 0 or d_out < 0:       # angka baru saja di-reset dari dashboard
                continue
            for _ in range(min(d_in, 20)):
                record_event("in", source="gate", camera_id=cam_id, camera_name=worker.name,
                             gate="Gerbang Masuk" if "in" in kinds else worker.name)
            for _ in range(min(d_out, 20)):
                record_event("out", source="gate", camera_id=cam_id, camera_name=worker.name,
                             gate="Gerbang Keluar" if "out" in kinds else worker.name)

    def _vote(self, slot_id: str, candidate: str, camera_id: int, camera_name: str):
        current = self._current.get(slot_id, "free")
        if current in ("reserved", "illegal"):      # dikunci operator, AI tidak menimpa
            return
        if candidate == current:
            self._pending.pop(slot_id, None)
            return

        status, count = self._pending.get(slot_id, (candidate, 0))
        if status != candidate:
            self._pending[slot_id] = (candidate, 1)
            return
        count += 1
        if count < PARKING_CONFIRM_FRAMES:
            self._pending[slot_id] = (candidate, count)
            return

        self._pending.pop(slot_id, None)
        self._current[slot_id] = candidate
        _write_status(slot_id, candidate)
        record_event("in" if candidate == "occupied" else "out", source="slot",
                     camera_id=camera_id, camera_name=camera_name, gate=camera_name,
                     slot_id=slot_id, vehicle_class="car")
        if _nx_alert:
            try:
                worker = _cam_mgr.get(camera_id) if _cam_mgr else None
                nx_id = getattr(worker, "nx_camera_id", "") if worker else ""
                verb = "terisi" if candidate == "occupied" else "kosong"
                _nx_alert(f"🅿️ Slot {slot_id} {verb} — {camera_name}",
                          f"Perubahan okupansi terkonfirmasi setelah {PARKING_CONFIRM_FRAMES} siklus "
                          f"({PARKING_CONFIRM_FRAMES * PARKING_POLL_SEC:.0f} detik).",
                          nx_id, tags=["parking", "AI_Alert"])
            except Exception as e:
                print(f"[Parking NX ERR] {e}")


watcher = ParkingWatcher()
tracker = watcher      # nama lama tetap dikenali