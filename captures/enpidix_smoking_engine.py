"""
enpidix_smoking_engine.py
======================================================================
ENPIDIX VMS — Deteksi merokok berbasis MULTI-BUKTI + penangkapan bukti.

KENAPA VERSI LAMA TERBALIK
--------------------------
Versi lama memakai SMOKING_PROXY_CLASSES = {"cell phone", "cup", "bottle"}:
kalau YOLO melihat salah satu benda itu di zona kepala, orangnya dianggap
merokok. Masalahnya bukan sekadar "kurang akurat" — logikanya TERBALIK:

  * Rokok lebarnya sekitar 8 mm. Pada frame 480-640 px dengan orang berdiri
    3-5 m dari kamera, rokok itu hanya beberapa piksel. YOLOv8n tidak akan
    pernah mengklasifikasikannya sebagai "cup" atau "bottle" atau apapun.
    Jadi PEROKOK ASLI hampir selalu LOLOS.
  * Sebaliknya, orang menelepon (cell phone di dekat wajah) dan orang minum
    kopi (cup di dekat wajah) justru terdeteksi hampir setiap kali.

Hasil bersihnya: yang ditangkap orang yang menelepon dan minum, yang merokok
lolos. Menaikkan SMOKING_CONFIRM_FRAMES tidak menolong — konfirmasi berulang
justru MEMPERKUAT false positive, karena orang menelepon memegang HP-nya lama
sekali sementara perokok mengangkat tangan cuma beberapa detik.

PENDEKATAN BARU: 4 BUKTI YANG DINILAI TERPISAH
----------------------------------------------
B1 GESTUR SIKLIK — bukti terkuat, dan satu-satunya yang benar-benar memisahkan
   merokok dari menelepon. Pola merokok punya tanda tangan waktu yang khas:
   tangan naik ke mulut, DITAHAN 1-4 detik (satu isapan), lalu TURUN dan pergi
   10-40 detik, berulang. Menelepon = tangan naik lalu DIAM DI ATAS berpuluh
   detik tanpa siklus. Minum = satu-dua kali angkat lalu gelasnya pergi.
   Jadi yang dihitung bukan "ada tangan di dekat mulut", melainkan JUMLAH SIKLUS
   naik-turun yang durasi tahannya masuk rentang isapan. Dwell yang terlalu
   lama justru MENGURANGI skor, bukan menambah.

B2 KEPULAN ASAP DI SEKITAR KEPALA — gumpalan bersaturasi rendah yang MUNCUL
   LALU HILANG di atas kepala. Dinding abu-abu tidak lolos karena syaratnya
   harus ada perubahan temporal, bukan sekadar warna abu-abu.

B3 BARA API — titik kecil terang berwarna hangat di zona mulut/tangan. Hanya
   dinilai saat cahaya sekitar redup; siang hari isyarat ini tidak dipakai
   karena pantulan matahari terlalu sering menirunya.

B4 PENYANGKAL (dibalik dari versi lama) — kalau YOLO memang melihat cell phone
   / cup / bottle / wine glass di zona kepala, itu sekarang MENGURANGI skor.
   Inilah pembalikan logika yang paling menentukan.

Skor akhir = gabungan berbobot, dengan syarat tambahan minimal satu bukti kuat.

PENANGKAPAN BUKTI
-----------------
Alert lama menyimpan satu potongan kotak orang dengan label "dekat cell phone" —
tidak bisa dipakai sebagai bukti apapun. Modul ini menyimpan:
  * pre-roll: cuplikan beberapa detik SEBELUM alert (ring buffer), karena isapan
    yang memicu alert selalu sudah terjadi saat alert menyala
  * frame penuh beranotasi (kotak orang + skor)
  * crop orang, dan crop kepala yang diperbesar
  * berkas JSON berisi rincian tiap bukti + waktu tiap isapan, supaya alert bisa
    dipertanggungjawabkan, bukan cuma "indikasi merokok"

MODEL POSE (OPSIONAL, SANGAT DIREKOMENDASIKAN)
----------------------------------------------
Kalau yolov8n-pose.pt tersedia, B1 memakai titik tubuh asli (pergelangan tangan
& hidung) sehingga siklus isapan terukur langsung. Tanpa model itu, B1 memakai
perkiraan berbasis gerakan yang JAUH lebih lemah — tanpa pose, keuntungan utama
modul ini adalah menghapus false positive (B4), bukan menangkap perokok asli.
Model pose ~6.5 MB dan memakai ultralytics yang sudah terpasang.
======================================================================
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# =====================================================================
# KONFIGURASI
# =====================================================================

POSE_MODEL_PATH = os.environ.get("SMOKING_POSE_MODEL", "models/yolov8n-pose.pt")
POSE_ENABLED = os.environ.get("SMOKING_POSE_ENABLED", "1") != "0"

# --- B1: gestur siklik ---
# Rentang durasi satu isapan. Di bawah batas bawah biasanya cuma tangan lewat
# (menggaruk, membetulkan masker). Di atas batas atas hampir pasti menelepon.
DRAG_MIN_SEC = float(os.environ.get("SMOKING_DRAG_MIN_SEC", "0.6"))
DRAG_MAX_SEC = float(os.environ.get("SMOKING_DRAG_MAX_SEC", "4.5"))
# Dwell selama ini tanpa turun = pola menelepon -> hukuman.
PHONE_DWELL_SEC = float(os.environ.get("SMOKING_PHONE_DWELL_SEC", "8.0"))
# Jendela pengamatan & jumlah isapan minimum di dalamnya.
WINDOW_SEC = float(os.environ.get("SMOKING_WINDOW_SEC", "50"))
MIN_CYCLES = int(os.environ.get("SMOKING_MIN_CYCLES", "2"))
# Jarak pergelangan-ke-mulut (dinormalkan lebar bahu) yang dianggap "di mulut".
WRIST_NEAR = float(os.environ.get("SMOKING_WRIST_NEAR", "0.62"))

# --- B2: kepulan asap ---
SMOKE_SAT_MAX = int(os.environ.get("SMOKING_SMOKE_SAT_MAX", "62"))
SMOKE_VAL_MIN = int(os.environ.get("SMOKING_SMOKE_VAL_MIN", "80"))
SMOKE_VAL_MAX = int(os.environ.get("SMOKING_SMOKE_VAL_MAX", "225"))
SMOKE_AREA_PCT = float(os.environ.get("SMOKING_SMOKE_AREA_PCT", "3.0"))
SMOKE_DIFF_MIN = int(os.environ.get("SMOKING_SMOKE_DIFF_MIN", "12"))

# --- B3: bara ---
EMBER_DARK_MAX = int(os.environ.get("SMOKING_EMBER_DARK_MAX", "110"))  # baru dinilai jika lebih gelap dari ini
EMBER_AREA_MIN = int(os.environ.get("SMOKING_EMBER_AREA_MIN", "2"))
EMBER_AREA_MAX = int(os.environ.get("SMOKING_EMBER_AREA_MAX", "70"))

# --- B4: penyangkal ---
CONFUSER_CLASSES = {"cell phone", "cup", "bottle", "wine glass", "bowl", "sandwich", "donut"}
CONFUSER_PENALTY = float(os.environ.get("SMOKING_CONFUSER_PENALTY", "0.45"))

# --- Fusi & ambang ---
W_GESTURE = float(os.environ.get("SMOKING_W_GESTURE", "0.55"))
W_SMOKE = float(os.environ.get("SMOKING_W_SMOKE", "0.30"))
W_EMBER = float(os.environ.get("SMOKING_W_EMBER", "0.15"))
SCORE_THRESHOLD = float(os.environ.get("SMOKING_SCORE_THRESHOLD", "0.55"))
# Minimal satu bukti harus "kuat" sendirian. Tanpa syarat ini, tiga bukti lemah
# bisa menumpuk jadi alert padahal tak satupun benar-benar meyakinkan.
STRONG_CUE_MIN = float(os.environ.get("SMOKING_STRONG_CUE_MIN", "0.60"))
# Zona kepala: % tinggi teratas box orang yang dianggap area kepala/mulut.
HEAD_ZONE_PCT = float(os.environ.get("SMOKING_HEAD_ZONE_PCT", "28"))

# --- Pelacakan orang ---
TRACK_MAX_DIST_PCT = float(os.environ.get("SMOKING_TRACK_DIST_PCT", "18"))  # % lebar frame
TRACK_MAX_MISSED = int(os.environ.get("SMOKING_TRACK_MAX_MISSED", "25"))
ALERT_REPEAT_SEC = float(os.environ.get("SMOKING_ALERT_REPEAT_SEC", "120"))

# --- Penangkapan bukti ---
EVIDENCE_ENABLED = os.environ.get("SMOKING_EVIDENCE", "1") != "0"
PREROLL_SEC = float(os.environ.get("SMOKING_PREROLL_SEC", "8"))
PREROLL_FPS = float(os.environ.get("SMOKING_PREROLL_FPS", "3"))
PREROLL_WIDTH = int(os.environ.get("SMOKING_PREROLL_WIDTH", "640"))
PREROLL_QUALITY = int(os.environ.get("SMOKING_PREROLL_QUALITY", "70"))
POSTROLL_SEC = float(os.environ.get("SMOKING_POSTROLL_SEC", "4"))


# =====================================================================
# DETEKTOR BUKTI TINGKAT RENDAH
# =====================================================================

def _clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))


def head_zone(person_box) -> tuple:
    """Zona kepala/mulut: bagian teratas box orang, dipersempit horizontal.

    Dipersempit karena tepi kiri-kanan box orang berdiri biasanya berisi latar
    belakang, bukan tubuh — dan latar belakang itulah sumber false positive asap.
    """
    x1, y1, x2, y2 = person_box
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    return (int(x1 + 0.18 * w), int(y1),
            int(x2 - 0.18 * w), int(y1 + h * HEAD_ZONE_PCT / 100.0))


def mouth_zone(person_box) -> tuple:
    """Pita mulut: kira-kira sepertiga bawah kepala."""
    hx1, hy1, hx2, hy2 = head_zone(person_box)
    hh = max(1, hy2 - hy1)
    return (hx1, int(hy1 + 0.45 * hh), hx2, int(hy1 + 1.15 * hh))


def detect_smoke_plume(frame_bgr, prev_gray, person_box) -> float:
    """B2 — kepulan asap di sekitar & di atas kepala.

    Syaratnya dua-duanya harus terpenuhi: warna khas asap (saturasi rendah,
    kecerahan menengah) DAN berubah dibanding frame sebelumnya. Syarat perubahan
    itu yang membuat dinding abu-abu, langit mendung, dan seragam abu-abu tidak
    ikut terhitung — benda-benda itu diam, asap tidak.
    Return: 0..1
    """
    if prev_gray is None:
        return 0.0
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = person_box
    ph = max(1, y2 - y1)
    pw = max(1, x2 - x1)
    # Kotak pengamatan: dari sedikit di atas kepala sampai dagu, dilebarkan
    # karena asap menyebar ke samping.
    rx1, ry1, rx2, ry2 = _clip_box(
        (x1 - 0.25 * pw, y1 - 0.55 * ph, x2 + 0.25 * pw, y1 + 0.35 * ph), W, H)
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return 0.0

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, (0, 0, SMOKE_VAL_MIN), (180, SMOKE_SAT_MAX, SMOKE_VAL_MAX))

    cur_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[ry1:ry2, rx1:rx2]
    if prev_roi.shape != cur_gray.shape:
        return 0.0
    diff = cv2.absdiff(cur_gray, prev_roi)
    motion_mask = (diff > SMOKE_DIFF_MIN).astype(np.uint8) * 255

    mask = cv2.bitwise_and(color_mask, motion_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    area_pct = 100.0 * float(np.count_nonzero(mask)) / mask.size
    if area_pct < SMOKE_AREA_PCT:
        return 0.0
    # Naik landai: 1x ambang -> 0.4, 3x ambang -> 1.0
    return float(min(1.0, 0.4 + 0.3 * (area_pct / SMOKE_AREA_PCT - 1.0)))


def detect_ember(frame_bgr, person_box) -> float:
    """B3 — bara rokok: titik kecil terang berwarna hangat di zona mulut/tangan.

    Hanya dinilai kalau area orang tersebut memang redup. Di bawah cahaya
    terang, pantulan logam, kancing mengkilap, dan bintik matahari terlalu
    sering menyerupai bara sehingga isyarat ini lebih banyak merusak daripada
    menolong.
    Return: 0..1
    """
    H, W = frame_bgr.shape[:2]
    px1, py1, px2, py2 = _clip_box(person_box, W, H)
    if px2 - px1 < 8 or py2 - py1 < 8:
        return 0.0
    person_v = cv2.cvtColor(frame_bgr[py1:py2, px1:px2], cv2.COLOR_BGR2HSV)[:, :, 2]
    if float(np.mean(person_v)) > EMBER_DARK_MAX:
        return 0.0     # terlalu terang -> isyarat ini tidak dapat dipercaya

    mx1, my1, mx2, my2 = _clip_box(mouth_zone(person_box), W, H)
    if mx2 - mx1 < 6 or my2 - my1 < 6:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr[my1:my2, mx1:mx2], cv2.COLOR_BGR2HSV)
    warm = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 90, 205), (25, 255, 255)),
        cv2.inRange(hsv, (160, 90, 205), (180, 255, 255)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(warm, connectivity=8)
    best = 0.0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if EMBER_AREA_MIN <= a <= EMBER_AREA_MAX:
            best = max(best, min(1.0, 0.5 + a / (2.0 * EMBER_AREA_MAX)))
    return best


# =====================================================================
# BUKTI 1 — GESTUR SIKLIK
# =====================================================================

class GestureTracker:
    """Ubah rentetan status "tangan di mulut / tidak" menjadi jumlah ISAPAN.

    Yang dihitung transisi, bukan lamanya. Ini inti pemisah merokok vs
    menelepon: keduanya sama-sama "tangan di dekat mulut", tapi hanya merokok
    yang polanya naik-tahan-turun berulang.
    """

    def __init__(self):
        self.active = False
        self.since = 0.0
        self.cycles = deque()        # timestamp tiap isapan yang sah
        self.long_dwell = 0.0        # dwell terpanjang yang sedang berjalan (detik)
        self.last_dwell = 0.0

    def push(self, near: bool, ts: float):
        if near and not self.active:
            self.active, self.since = True, ts
        elif near and self.active:
            self.long_dwell = ts - self.since
        elif not near and self.active:
            dwell = ts - self.since
            self.active, self.long_dwell, self.last_dwell = False, 0.0, dwell
            if DRAG_MIN_SEC <= dwell <= DRAG_MAX_SEC:
                self.cycles.append(ts)
        while self.cycles and ts - self.cycles[0] > WINDOW_SEC:
            self.cycles.popleft()

    def score(self, ts: float) -> float:
        n = len(self.cycles)
        s = min(1.0, n / float(max(1, MIN_CYCLES)))
        # Tangan yang menetap lama di mulut = pola menelepon, bukan merokok.
        # Ini menekan skor, bukan menaikkannya seperti logika confirm-frames lama.
        dwell = self.long_dwell if self.active else 0.0
        if dwell > PHONE_DWELL_SEC:
            s *= max(0.0, 1.0 - (dwell - PHONE_DWELL_SEC) / PHONE_DWELL_SEC)
        return float(max(0.0, s))

    def info(self) -> dict:
        return {"isapan": len(self.cycles), "dwell_terakhir": round(self.last_dwell, 1),
                "dwell_berjalan": round(self.long_dwell, 1)}


# =====================================================================
# PEREKAM BARANG BUKTI
# =====================================================================

class EvidenceRecorder:
    """Ring buffer frame + penulis paket bukti.

    Alasan pre-roll: saat skor akhirnya melewati ambang, isapan yang membentuk
    bukti itu SUDAH SELESAI beberapa detik sebelumnya. Menyimpan hanya frame
    saat alert menyala berarti menyimpan gambar orang yang tangannya sudah
    turun — gambar yang justru tidak memperlihatkan apa-apa.
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.buf = deque()             # (ts, jpeg_bytes)
        self._last_push = 0.0
        self._pending = {}             # event_id -> dict paket yang masih menunggu post-roll

    def push_frame(self, frame_bgr, ts: float):
        if not EVIDENCE_ENABLED:
            return
        if ts - self._last_push < 1.0 / max(0.5, PREROLL_FPS):
            return                      # ambil sampel, bukan tiap frame -> hemat CPU
        self._last_push = ts
        h, w = frame_bgr.shape[:2]
        if w > PREROLL_WIDTH:
            sc = PREROLL_WIDTH / float(w)
            small = cv2.resize(frame_bgr, (PREROLL_WIDTH, int(h * sc)), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, PREROLL_QUALITY])
        if ok:
            self.buf.append((ts, buf.tobytes()))
        while self.buf and ts - self.buf[0][0] > PREROLL_SEC:
            self.buf.popleft()

        for ev in list(self._pending.values()):
            if ts <= ev["until"]:
                ev["post"].append((ts, buf.tobytes() if ok else b""))
            else:
                self._finish(ev)

    def open_case(self, frame_bgr, person_box, score: float, cues: dict,
                  camera_id, camera_name: str, ts: float) -> dict:
        """Buka satu paket bukti: pre-roll langsung disalin, post-roll dikumpulkan
        beberapa detik ke depan lalu paket ditutup otomatis."""
        eid = f"{datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')}_smoking_{uuid.uuid4().hex[:6]}"
        case_dir = os.path.join(self.out_dir, str(camera_id), eid)
        H, W = frame_bgr.shape[:2]

        try:
            os.makedirs(case_dir, exist_ok=True)

            # 1) frame penuh beranotasi
            ann = frame_bgr.copy()
            x1, y1, x2, y2 = _clip_box(person_box, W, H)
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 0, 255), 2)
            hx1, hy1, hx2, hy2 = _clip_box(head_zone(person_box), W, H)
            cv2.rectangle(ann, (hx1, hy1), (hx2, hy2), (0, 200, 255), 1)
            cv2.putText(ann, f"MEROKOK {score:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(case_dir, "frame_anotasi.jpg"), ann)

            # 2) crop orang
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size:
                cv2.imwrite(os.path.join(case_dir, "orang.jpg"), crop)

            # 3) crop kepala diperbesar 3x — ini yang biasanya dilihat manusia
            #    untuk memutuskan "benar rokok atau bukan"
            hc = frame_bgr[hy1:hy2, hx1:hx2]
            if hc.size:
                cv2.imwrite(os.path.join(case_dir, "kepala.jpg"),
                            cv2.resize(hc, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))

            # 4) pre-roll
            pre_dir = os.path.join(case_dir, "pra_kejadian")
            os.makedirs(pre_dir, exist_ok=True)
            for i, (bts, jb) in enumerate(list(self.buf)):
                with open(os.path.join(pre_dir, f"{i:02d}_{bts - ts:+.1f}s.jpg"), "wb") as f:
                    f.write(jb)
        except Exception as e:
            print(f"⚠️  [smoking] gagal menulis bukti: {e}")

        ev = {"id": eid, "dir": case_dir, "post": [], "until": ts + POSTROLL_SEC,
              "meta": {"event_id": eid, "tipe": "smoking", "skor": round(score, 3),
                       "bukti": cues, "camera_id": camera_id, "camera_name": camera_name,
                       "box_orang": [x1, y1, x2, y2],
                       "waktu": datetime.fromtimestamp(ts).isoformat(),
                       "pra_kejadian_detik": PREROLL_SEC}}
        self._pending[eid] = ev
        return ev

    def _finish(self, ev):
        try:
            if ev["post"]:
                post_dir = os.path.join(ev["dir"], "pasca_kejadian")
                os.makedirs(post_dir, exist_ok=True)
                t0 = ev["until"] - POSTROLL_SEC
                for i, (bts, jb) in enumerate(ev["post"]):
                    if jb:
                        with open(os.path.join(post_dir, f"{i:02d}_{bts - t0:+.1f}s.jpg"), "wb") as f:
                            f.write(jb)
            with open(os.path.join(ev["dir"], "bukti.json"), "w", encoding="utf-8") as f:
                json.dump(ev["meta"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  [smoking] gagal menutup paket bukti: {e}")
        self._pending.pop(ev["id"], None)

    def flush(self):
        for ev in list(self._pending.values()):
            self._finish(ev)


# =====================================================================
# ENGINE
# =====================================================================

class SmokingEngine:
    """Satu instance per kamera (disimpan di CameraWorker)."""

    def __init__(self, camera_id, camera_name: str, evidence_dir: str):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.tracks = {}          # tid -> state
        self.next_id = 1
        self.prev_gray = None
        self.recorder = EvidenceRecorder(evidence_dir)
        self._pose = None
        self._pose_tried = False

    # ---------- model pose opsional ----------
    def _pose_model(self):
        if not POSE_ENABLED or self._pose_tried:
            return self._pose
        self._pose_tried = True
        if not os.path.exists(POSE_MODEL_PATH):
            print(f"ℹ️  [smoking] {POSE_MODEL_PATH} tidak ada — bukti gestur memakai "
                  f"perkiraan gerakan yang jauh lebih lemah. Deteksi tetap jalan, "
                  f"tapi manfaat utamanya jadi sebatas menekan false positive.")
            return None
        try:
            from ultralytics import YOLO
            self._pose = YOLO(POSE_MODEL_PATH)
            print(f"✅ [smoking] model pose dimuat: {POSE_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  [smoking] gagal memuat model pose: {e}")
            self._pose = None
        return self._pose

    def pose_available(self) -> bool:
        return self._pose_model() is not None

    # ---------- gestur ----------
    def _wrist_near_mouth(self, frame_bgr, person_box, track) -> Optional[bool]:
        """True/False kalau bisa dinilai, None kalau tidak ada isyarat sama sekali."""
        model = self._pose_model()
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = _clip_box(person_box, W, H)
        if x2 - x1 < 24 or y2 - y1 < 48:
            return None

        if model is not None:
            try:
                crop = frame_bgr[y1:y2, x1:x2]
                res = model.predict(crop, imgsz=192, conf=0.35, verbose=False)
                if not res or res[0].keypoints is None or len(res[0].keypoints) == 0:
                    return None
                kp = res[0].keypoints.data[0].cpu().numpy()   # (17, 3)
                nose, lsh, rsh = kp[0], kp[5], kp[6]
                lw, rw = kp[9], kp[10]
                if nose[2] < 0.3 or lsh[2] < 0.3 or rsh[2] < 0.3:
                    return None
                shoulder = float(np.hypot(lsh[0] - rsh[0], lsh[1] - rsh[1]))
                if shoulder < 5:
                    return None
                # Mulut ~ sedikit di bawah hidung.
                mouth = np.array([nose[0], nose[1] + 0.18 * shoulder])
                best = 1e9
                for w_ in (lw, rw):
                    if w_[2] < 0.3:
                        continue
                    best = min(best, float(np.hypot(w_[0] - mouth[0], w_[1] - mouth[1])) / shoulder)
                if best > 1e8:
                    return None
                track["dbg_wrist"] = round(best, 2)
                return best < WRIST_NEAR
            except Exception:
                return None

        # --- Fallback tanpa model pose: energi gerak di pita mulut ---
        # Ini PERKIRAAN KASAR. Kepala menoleh juga menimbulkan gerakan di sini,
        # jadi jangan diperlakukan setara dengan pengukuran pose.
        mx1, my1, mx2, my2 = _clip_box(mouth_zone(person_box), W, H)
        if mx2 - mx1 < 6 or my2 - my1 < 6:
            return None
        cur = cv2.resize(cv2.cvtColor(frame_bgr[my1:my2, mx1:mx2], cv2.COLOR_BGR2GRAY), (24, 16))
        prev = track.get("mouth_prev")
        track["mouth_prev"] = cur
        if prev is None:
            return None
        energy = float(np.mean(cv2.absdiff(cur, prev)))
        track["dbg_energy"] = round(energy, 1)
        return energy > float(os.environ.get("SMOKING_MOUTH_ENERGY", "9"))

    # ---------- pelacakan ----------
    def _match(self, person_boxes, frame_w, ts):
        max_d = frame_w * TRACK_MAX_DIST_PCT / 100.0
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in person_boxes]
        unmatched = set(range(len(person_boxes)))

        for tid, tr in list(self.tracks.items()):
            best_i, best_d = None, max_d
            for i in unmatched:
                d = np.hypot(centers[i][0] - tr["cx"], centers[i][1] - tr["cy"])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is None:
                tr["missed"] += 1
                if tr["missed"] > TRACK_MAX_MISSED:
                    del self.tracks[tid]
                continue
            tr.update(cx=centers[best_i][0], cy=centers[best_i][1],
                      box=person_boxes[best_i], missed=0, last_ts=ts)
            tr["_idx"] = best_i
            unmatched.discard(best_i)

        for i in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "cx": centers[i][0], "cy": centers[i][1], "box": person_boxes[i],
                "missed": 0, "last_ts": ts, "_idx": i,
                "gesture": GestureTracker(), "smoke": deque(), "ember": deque(),
                "confuser": deque(), "mouth_prev": None, "alerted_at": 0.0,
                "score": 0.0,
            }
        return self.tracks

    # ---------- API utama ----------
    def update(self, frame_bgr, person_boxes: list, object_boxes: list,
               ts: Optional[float] = None) -> list:
        """Proses satu siklus.

        person_boxes : list (x1, y1, x2, y2) hasil YOLO class "person"
        object_boxes : list (x1, y1, x2, y2, cls_name) semua objek lain — dipakai
                       sebagai PENYANGKAL, bukan sebagai bukti merokok.
        Return: list event alert (biasanya kosong).
        """
        ts = ts or time.time()
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.recorder.push_frame(frame_bgr, ts)

        events = []
        if not person_boxes:
            self.prev_gray = gray
            for tr in self.tracks.values():
                tr["missed"] += 1
            return events

        confusers = [o for o in object_boxes if o[4] in CONFUSER_CLASSES]
        self._match(person_boxes, W, ts)

        for tid, tr in self.tracks.items():
            if tr["missed"] > 0:
                continue
            box = tr["box"]

            # --- B1 gestur ---
            near = self._wrist_near_mouth(frame_bgr, box, tr)
            if near is not None:
                tr["gesture"].push(bool(near), ts)
            g = tr["gesture"].score(ts)

            # --- B2 asap ---
            sm = detect_smoke_plume(frame_bgr, self.prev_gray, box)
            tr["smoke"].append((ts, sm))
            # --- B3 bara ---
            em = detect_ember(frame_bgr, box)
            tr["ember"].append((ts, em))
            # --- B4 penyangkal ---
            hx1, hy1, hx2, hy2 = head_zone(box)
            hit = any(not (ox2 < hx1 or ox1 > hx2 or oy2 < hy1 or oy1 > hy2)
                      for (ox1, oy1, ox2, oy2, _c) in confusers)
            tr["confuser"].append((ts, 1.0 if hit else 0.0))

            for key in ("smoke", "ember", "confuser"):
                while tr[key] and ts - tr[key][0][0] > WINDOW_SEC:
                    tr[key].popleft()

            s_smoke = float(np.mean(sorted((v for _t, v in tr["smoke"]), reverse=True)[:5])) \
                if tr["smoke"] else 0.0
            s_ember = float(np.mean(sorted((v for _t, v in tr["ember"]), reverse=True)[:5])) \
                if tr["ember"] else 0.0
            c_frac = float(np.mean([v for _t, v in tr["confuser"]])) if tr["confuser"] else 0.0

            raw = (W_GESTURE * g + W_SMOKE * s_smoke + W_EMBER * s_ember) / \
                  (W_GESTURE + W_SMOKE + W_EMBER)
            score = max(0.0, raw - CONFUSER_PENALTY * c_frac)
            tr["score"] = score

            strong = max(g, s_smoke, s_ember) >= STRONG_CUE_MIN
            cooled = ts - tr["alerted_at"] >= ALERT_REPEAT_SEC

            if score >= SCORE_THRESHOLD and strong and cooled:
                tr["alerted_at"] = ts
                cues = {
                    "gestur_siklik": round(g, 3), **tr["gesture"].info(),
                    "kepulan_asap": round(s_smoke, 3),
                    "bara": round(s_ember, 3),
                    "penyangkal_terlihat": round(c_frac, 3),
                    "sumber_gestur": "pose" if self.pose_available() else "perkiraan_gerakan",
                }
                case = self.recorder.open_case(frame_bgr, box, score, cues,
                                               self.camera_id, self.camera_name, ts) \
                    if EVIDENCE_ENABLED else {"id": None, "dir": None}
                events.append({
                    "track_id": tid, "box": box, "score": round(score, 3),
                    "cues": cues, "evidence_id": case["id"], "evidence_dir": case["dir"],
                    "label": f"Merokok ({score:.0%})",
                })

        self.prev_gray = gray
        return events

    def snapshot(self) -> list:
        """Kondisi tiap orang yang sedang dipantau — untuk overlay & debug."""
        out = []
        for tid, tr in self.tracks.items():
            if tr["missed"] > 3:
                continue
            out.append({"track_id": tid, "box": tr["box"], "score": round(tr["score"], 3),
                        **tr["gesture"].info()})
        return out

    def close(self):
        self.recorder.flush()
