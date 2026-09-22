#!/usr/bin/env bash
# =============================================================================
# Setup AI Surveillance VMS Server di NVIDIA Jetson Orin NX
# =============================================================================
# Jalankan LANGSUNG di board Orin NX (bukan di PC/laptop), setelah JetPack
# ter-flash. Cek dulu versi JetPack yang terpasang:
#     sudo apt-cache show nvidia-jetpack | grep Version
#
# Script ini:
#   1. Pasang dependency sistem (ffmpeg, build tools, OpenCV system deps)
#   2. Pasang PyTorch + torchvision build KHUSUS Jetson (wheel PyPI biasa
#      TIDAK punya dukungan CUDA untuk ARM+Jetson -- harus dari NVIDIA)
#   3. Pasang ultralytics, onvif-zeep, psutil, jetson-stats (jtop), dll
#   4. Nyalakan jtop.service (dibutuhkan server buat baca GPU load/suhu/power)
#   5. Set power mode ke MAXN + jetson_clocks (performa maksimum, bukan mode
#      hemat daya default) -- WAJIB supaya YOLO benar-benar dapat clock penuh
#   6. Tes cepat: pastikan torch.cuda.is_available() == True
#
# Setelah script ini selesai, jalankan server dengan:
#     cd /path/ke/project && python3 server.py
# atau pasang sebagai service permanen, lihat systemd/enpidix-vms.service
# =============================================================================
set -e

echo "🚀 [1/6] Update & pasang dependency sistem..."
sudo apt update
sudo apt install -y \
    python3-pip python3-dev python3-venv \
    ffmpeg \
    libopenblas-dev libopenmpi-dev \
    libjpeg-dev zlib1g-dev \
    curl

echo ""
echo "🧠 [2/6] Cek JetPack yang terpasang..."
JETPACK_VER=$(apt-cache show nvidia-jetpack 2>/dev/null | grep -m1 "Version" | awk '{print $2}' || echo "unknown")
echo "   JetPack terdeteksi: ${JETPACK_VER}"
echo "   ⚠️  Pastikan wheel torch di bawah cocok dengan JetPack ini. Kalau versi"
echo "      berbeda, cek wheel yang sesuai di:"
echo "      https://developer.nvidia.com/embedded/downloads#?search=PyTorch"
echo "      atau (lebih gampang) pakai dukungan resmi NVIDIA lewat pip index:"
echo "      https://pypi.jetson-ai-lab.io/  (jetson-ai-lab pip server, auto match JetPack)"

echo ""
echo "🔥 [3/6] Pasang PyTorch + torchvision build Jetson (via jetson-ai-lab pip index)..."
# jetson-ai-lab menyediakan wheel torch/torchvision yang sudah dikompilasi utk Jetson
# dan otomatis dicocokkan ke JetPack yang terdeteksi -- jauh lebih simpel daripada
# download manual .whl dari NVIDIA per versi.
pip3 install --break-system-packages --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch torchvision \
    || echo "   ⚠️  Auto-install torch Jetson gagal -- pasang manual sesuai JetPack Anda, lihat link di atas."

echo ""
echo "📦 [4/6] Pasang dependency Python project..."
pip3 install --break-system-packages -r "$(dirname "$0")/../requirements-jetson.txt"

echo ""
echo "📊 [5/6] Pasang & nyalakan jetson-stats (jtop) -- dibutuhkan server untuk"
echo "         baca GPU load / suhu / power draw (GPUtil/nvidia-smi TIDAK ada di Jetson)..."
sudo pip3 install -U jetson-stats
sudo systemctl enable jtop.service
sudo systemctl restart jtop.service
echo "   ⚠️  User yang menjalankan server HARUS jadi anggota grup 'jtop':"
echo "       sudo usermod -aG jtop \$USER   (lalu logout/login ulang)"

echo ""
echo "⚡ [6/6] Set power mode ke performa maksimum..."
sudo nvpmodel -m 0   # mode 0 = MAXN pada kebanyakan Orin NX (cek: sudo nvpmodel -q --verbose)
sudo jetson_clocks
echo "   Power mode aktif: $(sudo nvpmodel -q | head -1)"

echo ""
echo "🧪 Tes cepat CUDA..."
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA tersedia:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"

echo ""
echo "✅ Setup selesai."
echo "   Langkah selanjutnya (opsional tapi SANGAT disarankan untuk performa):"
echo "   -> Export model ke TensorRT engine: bash scripts/export_tensorrt_jetson.sh"
echo "   -> Jalankan server: python3 server.py"
echo "   -> Atau pasang sebagai service permanen: lihat systemd/enpidix-vms.service"
