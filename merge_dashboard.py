#!/usr/bin/env python3
"""
Menggabungkan modul Fleet ke dalam dashboard.html yang sudah ada.

Sengaja dibuat sebagai skrip, bukan file hasil sekali jadi: dashboard Anda
masih berkembang. Setiap kali dashboard.html berubah, jalankan ulang ini dan
Fleet ikut tergabung lagi — tidak perlu menyalin-tempel manual.

    python3 scripts/merge_dashboard.py <dashboard.html asli> [-o keluaran.html]

Idempoten: menjalankan dua kali tidak menggandakan apa pun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SNIP = Path(__file__).resolve().parent.parent / "client-monitor" / "_snippets"
MARK = "ENPIDIX-FLEET-MERGED"


def read(name: str) -> str:
    p = SNIP / name
    if not p.exists():
        sys.exit(f"❌ Snippet hilang: {p}")
    return p.read_text(encoding="utf-8")


def insert_before(html: str, anchor: str, block: str, label: str) -> str:
    i = html.find(anchor)
    if i == -1:
        sys.exit(f"❌ Anchor untuk {label} tidak ditemukan: {anchor!r}\n"
                 f"   Struktur dashboard berubah — sesuaikan anchor di skrip ini.")
    return html[:i] + block + html[i:]


def insert_after(html: str, anchor: str, block: str, label: str) -> str:
    i = html.find(anchor)
    if i == -1:
        sys.exit(f"❌ Anchor untuk {label} tidak ditemukan: {anchor!r}")
    j = i + len(anchor)
    return html[:j] + block + html[j:]


def merge(html: str) -> str:
    if MARK in html:
        print("ℹ️  Dashboard sudah tergabung sebelumnya — tidak ada perubahan.")
        return html

    # 1) CSS Fleet, tepat sebelum </style> pertama
    html = insert_before(
        html, "</style>",
        f"\n/* ===== {MARK} ===== */\n" + read("fleet.css") + "\n",
        "CSS",
    )

    # 2) Saklar lingkup Site/Fleet di topbar, setelah segmen mode
    html = insert_after(
        html,
        '<button data-mode="complex" class="active">Complex</button>\n    </div>',
        "\n\n" + read("fleet_scope.html").rstrip(),
        "saklar lingkup",
    )

    # 3) View Fleet, setelah blok .layout ditutup (tetap di dalam #app)
    html = insert_after(
        html,
        "    </aside>\n  </div>\n</div>",
        "\n\n" + read("fleet_view.html").rstrip(),
        "view fleet",
    )

    # 4) Tab Settings baru, setelah tab Penyimpanan
    html = insert_after(
        html,
        '<div class="settings-tab" data-tab="storage">💾 Penyimpanan</div>',
        "\n" + read("fleet_tab.html").rstrip(),
        "tab settings",
    )

    # 5) Section Settings, tepat sebelum penutup wadah section
    anchor_sec = '<section class="settings-section" id="tab-storage">'
    i = html.find(anchor_sec)
    if i == -1:
        sys.exit("❌ Section tab-storage tidak ditemukan.")
    end = html.find("</section>", i)
    if end == -1:
        sys.exit("❌ Penutup section tab-storage tidak ditemukan.")
    end += len("</section>")
    html = html[:end] + "\n\n" + read("fleet_settings.html").rstrip() + html[end:]

    # 6) JS Fleet sebagai blok <script> terpisah sebelum </html>
    js = f"\n<script>\n/* ===== {MARK} ===== */\n" + read("fleet.js") + "</script>\n"
    if "</body>" in html:
        html = insert_before(html, "</body>", js, "script")
    else:
        html = insert_before(html, "</html>", js, "script")

    return html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="dashboard.html asli")
    ap.add_argument("-o", "--output", default="dashboard.merged.html")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        sys.exit(f"❌ Tidak ditemukan: {src}")

    original = src.read_text(encoding="utf-8")
    merged = merge(original)
    Path(args.output).write_text(merged, encoding="utf-8")

    print(f"✅ {args.output}")
    print(f"   {len(original.splitlines()):>6} baris  →  {len(merged.splitlines()):>6} baris "
          f"(+{len(merged.splitlines()) - len(original.splitlines())})")
    print("\nLangkah berikutnya:")
    print("  1. Salin hasilnya menimpa dashboard.html di server site")
    print("  2. Salin edge-site/enpidix_fleet.py ke folder yang sama dengan server.py")
    print("  3. Tambahkan tiga baris pemasangan (lihat README)")


if __name__ == "__main__":
    main()
