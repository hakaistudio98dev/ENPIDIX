#!/usr/bin/env bash
# =============================================================================
# Export yolov8n.pt -> yolov8n.engine (TensorRT) LANGSUNG DI Orin NX ini
# =============================================================================
# PENTING -- baca dulu:
#   File .engine hasil export SPESIFIK untuk kombinasi (GPU + versi TensorRT +
#   versi JetPack) board yang dipakai saat export. TIDAK BISA disalin dari PC,
#   dari Orin NX lain, atau dari Orin Nano/AGX -- harus di-export ULANG di
#   board target masing-masing, sekali saja (hasilnya dipakai terus sampai
#   ganti board/JetPack).
#
# Setelah file yolov8n.engine muncul di folder project ini, server.py akan
# OTOMATIS memakainya saat restart (lihat YOLO_MODEL_PATH_DEFAULT di server.py)
# -- tidak perlu ubah kode apa pun.
#
# imgsz HARUS sama dengan YOLO_IMG_SIZE yang dipakai server (default 640 di
# Jetson dengan GPU aktif) -- engine TensorRT dikompilasi untuk resolusi input
# yang FIXED, tidak bisa dipakai untuk imgsz lain tanpa export ulang.
# =============================================================================
set -e

MODEL_SRC="${1:-yolov8n.pt}"
IMG_SIZE="${2:-640}"

echo "🔧 Export ${MODEL_SRC} -> TensorRT engine (imgsz=${IMG_SIZE}, FP16)..."
echo "   Ini bisa makan waktu 5-15 menit di Orin NX, tunggu sampai selesai."

yolo export model="${MODEL_SRC}" format=engine device=0 half=True imgsz="${IMG_SIZE}" workspace=4

ENGINE_FILE="${MODEL_SRC%.pt}.engine"
if [ -f "${ENGINE_FILE}" ]; then
    if [ "${ENGINE_FILE}" != "yolov8n.engine" ]; then
        mv "${ENGINE_FILE}" yolov8n.engine
    fi
    echo "✅ Berhasil: yolov8n.engine siap dipakai."
    echo "   Restart server (systemctl restart enpidix-vms, atau python3 server.py ulang)"
    echo "   supaya engine ini otomatis terpakai."
else
    echo "❌ Export tampaknya gagal -- file ${ENGINE_FILE} tidak ditemukan."
    echo "   Cek pesan error 'yolo export' di atas."
    exit 1
fi
