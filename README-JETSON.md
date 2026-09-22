# Menjalankan AI Surveillance VMS Server di NVIDIA Jetson Orin NX

Ini adalah `server.py` yang sudah diadaptasi supaya jalan optimal di Orin NX
(bukan cuma "bisa jalan", tapi benar-benar memakai GPU-nya). Perubahan dari
versi aslinya ada di 4 titik, semua otomatis — tidak perlu ubah kode:

1. **Deteksi hardware** — server membaca `/proc/device-tree/model` saat start
   untuk tahu apakah dirinya jalan di Jetson, dan cek `torch.cuda.is_available()`
   untuk tahu apakah CUDA benar-benar terbaca.
2. **Mode "low power" tidak lagi salah kaprah di Jetson** — sebelumnya server
   cuma tahu "CPU lemah = throttle semua". Sekarang Jetson dengan GPU aktif
   diberi profil AGRESIF (imgsz 640, skip frame lebih jarang, target FPS lebih
   tinggi) karena beban berat ada di GPU, bukan CPU ARM-nya.
3. **Model YOLO otomatis load ke GPU**, dan otomatis pakai file `yolov8n.engine`
   (TensorRT) kalau file itu ada di folder project — jauh lebih cepat & hemat
   memori daripada `.pt` biasa.
4. **Statistik GPU/suhu/power Jetson** ikut muncul di `/api/stats/system` dan
   `/health` lewat `jetson-stats` (jtop) — `GPUtil`/`nvidia-smi` yang dipakai
   kode asli TIDAK jalan di Jetson (GPU-nya bukan discrete GPU biasa).

Semua ini **fallback aman**: kalau dijalankan bukan di Jetson (laptop dev
biasa), perilaku persis seperti sebelumnya.

## Langkah instalasi

```bash
# 1) Di board Orin NX, clone/copy project ini, lalu:
cd enpidix-vms-server
bash scripts/setup_jetson_orin_nx.sh

# 2) Siapkan .env
cp .env.example .env
nano .env   # isi VMS_USERNAME / VMS_PASSWORD / VMS_SESSION_SECRET minimal

# 3) (SANGAT disarankan) export model ke TensorRT sekali saja
bash scripts/export_tensorrt_jetson.sh

# 4) Jalankan
set -a && source .env && set +a
python3 server.py
```

Cek `/health` — kalau setup benar, responnya harus menunjukkan
`"cuda": true` dan `"yolo_device": "cuda:0"`.

## Jalankan permanen (service, restart otomatis saat reboot)

```bash
sudo cp systemd/enpidix-vms.service /etc/systemd/system/
sudo nano /etc/systemd/system/enpidix-vms.service   # sesuaikan User & path
sudo systemctl daemon-reload
sudo systemctl enable --now enpidix-vms.service
journalctl -u enpidix-vms.service -f    # lihat log live
```

## Berapa kamera yang realistis di Orin NX?

Ini tergantung berapa fitur AI yang aktif BERSAMAAN per kamera. Server ini
punya banyak fitur (face recognition, smoking/fire/behavior heuristic, people
+ vehicle + animal counting, parking) yang semuanya jalan lewat **1x
inferensi YOLO per siklus** (`run_yolo` di `server.py`) — jadi biaya utamanya
bukan "jumlah fitur", tapi jumlah kamera × frekuensi inferensi.

| Konfigurasi | Perkiraan kamera 1080p tanpa lag |
|---|---|
| `.pt` biasa (belum export TensorRT) | ~4-6 kamera |
| `.engine` TensorRT (setelah `export_tensorrt_jetson.sh`) | ~8-10 kamera |
| `.engine` + turunkan `YOLO_IMG_SIZE=480` | ~10-14 kamera |
| `.engine` + `YOLO_EVERY_N_FRAMES=8` (analitik tiap ~0.4s, bukan tiap ~0.2s) | naik lagi ~1.3-1.5x |

Kalau kamera >10 dan mulai lag, urutan yang paling efektif dicoba (dari
dampak terbesar): pastikan pakai `.engine` bukan `.pt` → turunkan
`YOLO_IMG_SIZE` → naikkan `YOLO_EVERY_N_FRAMES` → matikan
`ai_face_recognition` di kamera yang tidak butuh (Haar cascade-nya jalan di
CPU, bukan GPU, jadi ini murni beban CPU terpisah dari YOLO).

Pantau langsung lewat `/api/stats/system` (field `ai.cameras[].fps` dan
`ai.cameras[].yolo_ms_avg` per kamera, plus `gpu[0].load_pct` dan
`jetson.power_draw_w`) untuk tahu kapan sudah mendekati batas, bukan
menebak-nebak.

## Hal yang PERLU diperhatikan manual (tidak bisa diotomasi dari kode)

- **Storage**: microSD/eMMC Jetson lambat & cepat aus untuk NVR 24/7. Arahkan
  `RECORDINGS_DIR`/`CAPTURES_DIR` ke SSD NVMe eksternal via M.2.
- **Power supply**: Orin NX di MAXN + banyak stream RTSP butuh adaptor sesuai
  spesifikasi carrier board (jangan pakai charger HP/laptop seadanya — bisa
  brown-out saat GPU spike, menyebabkan kamera "putus-putus" acak).
- **Pendinginan**: `jetson_clocks` mengunci clock di maksimum terus-menerus.
  Tanpa heatsink+fan yang memadai, board bisa thermal-throttle setelah
  beberapa jam jalan penuh — pantau `jetson.power_draw_w` dan suhu di jtop.
- **File `.engine` tidak portable** — kalau ganti board Orin NX atau upgrade
  JetPack, harus `export_tensorrt_jetson.sh` ulang di board yang baru.
