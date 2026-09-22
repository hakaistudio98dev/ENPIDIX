# ENPIDIX VMS — Solusi Terpadu

Satu sistem, tiga tier, **satu dashboard**.

```
client-monitor/    Dashboard tunggal — Site Ini + Fleet Nasional
regional-server/   Tier 2 — pengumpul event & proxy, 1 per wilayah
edge-site/         Tier 1 — modul yang menyatukan server.py Anda ke fleet
edge-jetson/       Tier 1 alternatif — edge node bersih tanpa server.py
common/            Kontrak data bersama
scripts/           Skrip merge dashboard & dev runner
docs/              Catatan arsitektur rinci
```

---

## Bagaimana penggabungannya bekerja

`server.py` Anda sudah berisi Live Grid, Playback, 3D Map, Dahua native AI,
counting, parkir, dan capture export. Semua itu **tetap utuh**. Yang berubah
hanya perannya: dari aplikasi tunggal per lokasi menjadi **edge node** dalam
jaringan nasional.

Penggabungan lewat dua jalur, keduanya aditif:

**Backend** — `edge-site/enpidix_fleet.py` dipasang dengan tiga baris, pola
yang sama seperti modul parkir. Tidak ada satu pun baris logika deteksi yang
disentuh.

**Frontend** — `scripts/merge_dashboard.py` menyisipkan bagian Fleet ke
`dashboard.html` Anda di lima titik. Idempoten dan bisa dijalankan ulang setiap
kali dashboard berkembang, jadi tidak ada salin-tempel manual yang berulang.

### Yang paling menghemat kerja

Setiap event AI di server.py sudah melewati `kirim_event_ke_nx`. Modul fleet
membungkus fungsi itu, sehingga **seluruh deteksi lama — api, merokok, wajah,
counting, Dahua native — otomatis ikut mengalir ke regional** tanpa mengubah
satu baris pun di jalur deteksi.

---

## Pemasangan

### 1. Backend site

```bash
cp edge-site/enpidix_fleet.py /path/ke/folder/server.py/
```

Tambahkan tiga baris di `server.py`:

```python
# (1) setelah `app = FastAPI(...)` — sekitar baris 139
import enpidix_fleet

# (2) tepat sebelum blok `if __name__ == "__main__":`
enpidix_fleet.install(app, globals())

# (3) di dalam @app.on_event("shutdown") yang sudah ada — sekitar baris 2426
enpidix_fleet.shutdown()
```

### 2. Dashboard

```bash
python3 scripts/merge_dashboard.py dashboard.html -o dashboard.new.html
# periksa hasilnya, lalu timpa
mv dashboard.new.html dashboard.html
```

### 3. Server regional

```bash
cd regional-server && pip install -r requirements.txt
cp regional.env.example /etc/enpidix/regional.env   # isi kedua API key
set -a && source /etc/enpidix/regional.env && set +a
uvicorn regional_main:app --host 0.0.0.0 --port 8200
```

### 4. Sambungkan

Dashboard site → **⚙ Settings → 🛰 Fleet & Regional** → isi Engine ID, Site ID,
URL regional, API key → centang aktifkan → **Tes Koneksi**.

Lalu **🌐 Fleet** → tab **Server Regional** → daftarkan regional → tab
**AI Engine** → site Anda sudah muncul sendiri dari heartbeat, tinggal
lengkapi host-nya.

---

## Cara memakai dashboard gabungan

Saklar di topbar menentukan lingkup:

**🏢 Site Ini** — persis seperti sebelumnya. Live grid, playback, parkir, Dahua
AI, counting, capture. Tidak ada yang berubah.

**🌐 Fleet** — pandangan nasional. Overview lintas wilayah, kartu status tiap
AI Engine dengan GPU/suhu/disk, registry engine, dan event log terkonsolidasi.
Tombol **Buka Dashboard Site** melompat ke dashboard lokal site tersebut.

---

## Patch performa

Bottleneck dari audit sebelumnya kini tersedia sebagai **saklar** di
Settings → Fleet, bukan perubahan permanen:

| Patch | Yang dilakukan | Titik sentuh |
|---|---|---|
| Decode NVDEC | GStreamer `nvv4l2decoder` menggantikan decode CPU | `CameraWorker._open_capture` |
| Rekaman stream copy | Membuang seluruh argumen encoding, ganti `-c copy` | `CameraRecorder._run_loop` |
| Inferensi paralel | Kunci global diganti slot terbatas | `yolo_lock` |

Ketiganya bisa dibatalkan tanpa restart, dan statusnya tampil di dashboard.

**Nyalakan satu per satu.** Menyalakan ketiganya sekaligus lalu menemukan
regresi berarti Anda tidak tahu patch mana penyebabnya. Ukur FPS dan CPU
sebelum dan sesudah tiap patch.

Dua hal yang mudah terlewat:

- **NVDEC butuh OpenCV bawaan JetPack.** Versi PyPI dibuild tanpa GStreamer.
  Kalau tidak tersedia, patch menolak menyala dan menyebutkan alasannya —
  tidak diam-diam jatuh ke CPU. Verifikasi:
  `python3 -c "import cv2;print(cv2.getBuildInformation())" | grep -i gstreamer`
- **Stream copy berlaku setelah rekaman di-restart**, karena perintah ffmpeg
  dibangun saat rekaman dimulai.

Bottleneck keempat — AI memakai main stream — **tidak bisa ditambal otomatis**
karena `server.py` menyimpan satu `rtsp_url` per kamera. Ini butuh kolom
`sub_rtsp_url` di tabel `cameras` dan perubahan di form kamera. Sampai itu
dikerjakan, isi `rtsp_url` dengan sub-stream dan pakai main stream hanya untuk
rekaman.

---

## Dua jalur edge

| | `edge-site/` | `edge-jetson/` |
|---|---|---|
| Basis | server.py Anda + modul fleet | Kode bersih dari nol |
| Fitur | Semua: parkir, 3D, Dahua, NX, playback | Inti: capture, infer, rekam, forward |
| Port | 8000 | 8100 |
| Untuk | Site existing & lokasi berfitur lengkap | Site baru minimalis, ratusan unit |

Keduanya berbicara protokol yang sama dan muncul berdampingan di dashboard
Fleet. Boleh dicampur dalam satu wilayah.

---

## Alur data

```
Kamera ──RTSP──▶ Site (server.py + fleet)
                   ├── event + heartbeat ──▶ Regional   ±0,4 Mbps, terus-menerus
                   └── live view          ◀── Regional   hanya saat ditonton
                                                 │
                              Dashboard ◀────────┘
```

Video tidak pernah naik dengan sendirinya. Menarik 375 stream ke satu regional
butuh ~1,9 Gbps; dengan pola ini cukup ~50 Mbps per wilayah.
