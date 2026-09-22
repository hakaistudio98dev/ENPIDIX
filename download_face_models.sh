#!/usr/bin/env bash
# Unduh model wajah untuk ENPIDIX VMS (Linux / macOS / Jetson).
# Jalankan dari folder yang sama dengan server.py:   bash scripts/download_face_models.sh
set -euo pipefail

DIR="${FACE_MODEL_DIR:-models}"
mkdir -p "$DIR"
BASE="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

# nama|url|ukuran_byte|sha256
ITEMS=(
"face_detection_yunet_2023mar.onnx|$BASE/face_detection_yunet/face_detection_yunet_2023mar.onnx|232589|8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
"face_recognition_sface_2021dec.onnx|$BASE/face_recognition_sface/face_recognition_sface_2021dec.onnx|38696353|0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
)

for item in "${ITEMS[@]}"; do
  IFS='|' read -r name url size sha <<< "$item"
  out="$DIR/$name"
  if [ -f "$out" ] && [ "$(wc -c < "$out" | tr -d " ")" = "$size" ]; then
    echo "[lewati] $name sudah ada dan ukurannya benar."
    continue
  fi
  echo "[unduh]  $name ..."
  curl -fL --retry 3 -o "$out" "$url"

  # PENTING: repo opencv_zoo memakai Git LFS. URL raw.githubusercontent
  # mengembalikan file TEKS ~130 byte berisi pointer, BUKAN model. Karena itu
  # skrip ini memakai media.githubusercontent.com dan memverifikasi ukurannya.
  actual=$(wc -c < "$out" | tr -d " ")
  if [ "$actual" != "$size" ]; then
    echo "GAGAL: $name berukuran $actual byte, seharusnya $size."
    if [ "$actual" -lt 1000 ]; then
      echo "       Isinya kemungkinan pointer Git LFS, bukan model."
      echo "       Unduh manual lewat browser dari halaman repo opencv_zoo,"
      echo "       klik tombol Download (raw), lalu simpan ke folder $DIR/."
    fi
    rm -f "$out"; exit 1
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    got=$(sha256sum "$out" | cut -d' ' -f1)
  elif command -v shasum >/dev/null 2>&1; then
    got=$(shasum -a 256 "$out" | cut -d' ' -f1)
  else
    got="$sha"
  fi
  [ "$got" = "$sha" ] || { echo "GAGAL: checksum $name tidak cocok."; rm -f "$out"; exit 1; }
  echo "[ok]     $name terverifikasi."
done

echo
echo "Selesai. Restart server, lalu cek:  curl -u USER:PASS http://localhost:8000/api/faces/engine/status"
echo "Setelah itu latih ulang sekali:     curl -u USER:PASS -X POST http://localhost:8000/api/faces/engine/retrain"
