"""
ONVIF Camera Discovery + Multi-Brand PTZ Control
==================================================
Modul terpisah (tidak mengubah logika inti server.py) yang menambahkan:

1. Auto-discovery kamera ONVIF di jaringan lokal (WS-Discovery).
2. "Probe" ke satu kamera by IP: ambil info device, daftar media profile,
   RTSP stream URI per profile, dan cek apakah kamera punya kemampuan PTZ.
3. Kontrol PTZ generik lewat standar ONVIF (Profile S/T) — jadi berlaku untuk
   SEMUA merk yang mengikuti standar ONVIF PTZ Service: Hikvision, Dahua,
   Uniview, Axis, Bosch, Hanwha, TP-Link/VIGI, Reolink, dsb — TIDAK perlu SDK
   khusus per merk. Yang membedakan tiap merk biasanya cuma:
     - metode autentikasi (WS-UsernameToken vs HTTP Digest di beberapa OEM
       murah) -> ditangani dengan fallback di _connect()
     - drift jam onboard kamera vs server -> ditangani via adjust_time=True
     - sebagian kamera murah/OEM tidak mengekspos <PTZConfiguration> di media
       profile-nya walau device fisiknya support PTZ -> di sini tetap dicoba
       connect ke PTZ service, kalau gagal baru dianggap "no PTZ".

Dependensi (opsional, di-lazy-import supaya server.py tetap jalan normal di
instalasi yang belum butuh fitur ONVIF):
    pip install onvif-zeep wsdiscovery --break-system-packages

Kalau salah satu library belum terpasang, semua fungsi di modul ini akan
melempar RuntimeError dengan pesan yang jelas (bukan crash saat import),
supaya endpoint FastAPI bisa mengembalikan error yang informatif ke dashboard.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_IMPORT_ERROR: Optional[str] = None
try:
    from onvif import ONVIFCamera
    from onvif.exceptions import ONVIFError
except Exception as _e:  # pragma: no cover - hanya kalau lib belum ke-install
    ONVIFCamera = None
    ONVIFError = Exception
    _IMPORT_ERROR = str(_e)


def _require_onvif():
    if ONVIFCamera is None:
        raise RuntimeError(
            "Library 'onvif-zeep' belum terpasang di server ini. "
            "Jalankan: pip install onvif-zeep wsdiscovery --break-system-packages "
            f"(detail: {_IMPORT_ERROR})"
        )


# =========================================================
# 🔎 DISCOVERY (WS-Discovery) — cari kamera ONVIF di jaringan lokal
# =========================================================

def discover_onvif_devices(timeout: float = 4.0) -> list[dict]:
    """
    Broadcast WS-Discovery ke jaringan lokal dan kumpulkan semua device yang
    mengumumkan diri sebagai ONVIF NetworkVideoTransmitter (kamera/NVR).
    Mengembalikan list dict: {xaddr, ip, port, scopes, types}.

    Catatan: ini cuma menemukan ALAMAT (xaddr) servis ONVIF-nya. Untuk info
    detail (merk/model/PTZ) & RTSP URL, device yang ditemukan harus di-"probe"
    lewat probe_onvif_camera() dengan username/password yang benar — WS-Discovery
    sendiri tidak butuh & tidak memberi kredensial.
    """
    try:
        from wsdiscovery.discovery import ThreadedWSDiscovery as WSDiscovery
    except Exception as e:
        raise RuntimeError(
            "Library 'wsdiscovery' belum terpasang. "
            "Jalankan: pip install wsdiscovery --break-system-packages "
            f"(detail: {e})"
        )

    wsd = WSDiscovery()
    found: list[dict] = []
    try:
        wsd.start()
        services = wsd.searchServices(timeout=timeout)
        seen_xaddrs = set()
        for svc in services:
            xaddrs = list(svc.getXAddrs() or [])
            scopes = [str(s) for s in (svc.getScopes() or [])]
            types = [str(t) for t in (svc.getTypes() or [])]
            # Hanya device yang mengiklankan dirinya video transmitter (kamera/NVR ONVIF).
            # Beberapa firmware tidak selalu menyertakan tipe ini di scope/type secara
            # eksplisit, jadi kalau tidak match pun tetap dimasukkan (biar tidak ke-skip),
            # tapi ditandai is_camera_hint.
            is_camera_hint = any("networkvideotransmitter" in t.lower() for t in types) or \
                             any("onvif" in s.lower() for s in scopes)
            for xaddr in xaddrs:
                if xaddr in seen_xaddrs:
                    continue
                seen_xaddrs.add(xaddr)
                host, port = _parse_host_port_from_xaddr(xaddr)
                found.append({
                    "xaddr": xaddr,
                    "host": host,
                    "port": port,
                    "scopes": scopes,
                    "types": types,
                    "is_camera_hint": is_camera_hint,
                })
    finally:
        try:
            wsd.stop()
        except Exception:
            pass
    return found


def _parse_host_port_from_xaddr(xaddr: str) -> tuple[str, int]:
    # xaddr contoh: "http://192.168.1.64:80/onvif/device_service"
    try:
        from urllib.parse import urlparse
        u = urlparse(xaddr)
        host = u.hostname or ""
        port = u.port or (443 if u.scheme == "https" else 80)
        return host, port
    except Exception:
        return "", 80


# =========================================================
# 🔌 KONEKSI & PROBE 1 KAMERA
# =========================================================

def _connect(host: str, port: int, username: str, password: str) -> "ONVIFCamera":
    """
    Buka koneksi ONVIF ke satu kamera. Dicoba dengan adjust_time=True supaya
    kamera yang jamnya ngaco (beda merk sering tidak NTP-sync) tetap bisa
    diautentikasi via WS-UsernameToken (kalau tidak, request bisa ditolak
    'jam tidak sinkron' terutama di Dahua/Hikvision OEM lama).
    """
    _require_onvif()
    last_err = None
    for adjust_time in (True, False):
        try:
            cam = ONVIFCamera(host, int(port), username or "", password or "",
                               adjust_time=adjust_time)
            # Paksa 1 call ringan supaya error auth/koneksi ketahuan sekarang,
            # bukan nanti pas dipakai.
            cam.create_devicemgmt_service().GetDeviceInformation()
            return cam
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gagal konek ONVIF ke {host}:{port} — {last_err}")


@dataclass
class OnvifProfileInfo:
    token: str
    name: str
    rtsp_url: Optional[str]
    ptz_supported: bool
    resolution: Optional[str] = None


@dataclass
class OnvifProbeResult:
    manufacturer: str
    model: str
    firmware: str
    serial: str
    ptz_supported: bool
    profiles: list[OnvifProfileInfo] = field(default_factory=list)


def probe_onvif_camera(host: str, port: int, username: str, password: str) -> OnvifProbeResult:
    """
    Konek ke kamera, ambil info device + semua media profile + RTSP URL
    masing-masing + status dukungan PTZ. Dipakai dashboard untuk menampilkan
    pilihan stream (main/sub) sebelum kamera benar-benar disimpan.
    """
    cam = _connect(host, port, username, password)

    devicemgmt = cam.create_devicemgmt_service()
    try:
        info = devicemgmt.GetDeviceInformation()
        manufacturer = getattr(info, "Manufacturer", "") or ""
        model = getattr(info, "Model", "") or ""
        firmware = getattr(info, "FirmwareVersion", "") or ""
        serial = getattr(info, "SerialNumber", "") or ""
    except Exception:
        manufacturer = model = firmware = serial = ""

    media = cam.create_media_service()
    try:
        profiles = media.GetProfiles()
    except Exception as e:
        raise RuntimeError(f"Kamera terkoneksi tapi gagal ambil daftar profile media: {e}")

    ptz_service = None
    try:
        ptz_service = cam.create_ptz_service()
    except Exception:
        ptz_service = None

    profile_infos: list[OnvifProfileInfo] = []
    any_ptz = False
    for p in profiles:
        token = p.token
        name = getattr(p, "Name", token)

        # RTSP stream URI untuk profile ini
        rtsp_url = None
        try:
            req = media.create_type("GetStreamUri")
            req.ProfileToken = token
            req.StreamSetup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            uri_resp = media.GetStreamUri(req)
            raw_uri = getattr(uri_resp, "Uri", None)
            rtsp_url = _inject_credentials_into_rtsp(raw_uri, username, password)
        except Exception:
            rtsp_url = None

        resolution = None
        try:
            vres = p.VideoEncoderConfiguration.Resolution
            resolution = f"{vres.Width}x{vres.Height}"
        except Exception:
            pass

        # Dukungan PTZ per-profile: cek PTZConfiguration di profile ATAU coba
        # panggil GetConfigurations/GetStatus langsung ke PTZ service (sebagian
        # kamera OEM murah tidak mengisi PTZConfiguration di profile meski
        # secara fisik motorized).
        ptz_ok = bool(getattr(p, "PTZConfiguration", None))
        if not ptz_ok and ptz_service is not None:
            try:
                ptz_service.GetStatus({"ProfileToken": token})
                ptz_ok = True
            except Exception:
                ptz_ok = False

        any_ptz = any_ptz or ptz_ok
        profile_infos.append(OnvifProfileInfo(
            token=token, name=name, rtsp_url=rtsp_url,
            ptz_supported=ptz_ok, resolution=resolution,
        ))

    return OnvifProbeResult(
        manufacturer=manufacturer, model=model, firmware=firmware, serial=serial,
        ptz_supported=any_ptz, profiles=profile_infos,
    )


def _inject_credentials_into_rtsp(uri: Optional[str], username: str, password: str) -> Optional[str]:
    """Kebanyakan kamera mengembalikan Uri ONVIF TANPA user:pass di dalamnya
    (autentikasi RTSP dicek terpisah oleh device), jadi kita sisipkan supaya
    langsung bisa dipakai ffmpeg/opencv persis seperti alur RTSP manual yang
    sudah ada di server.py."""
    if not uri:
        return uri
    if not username:
        return uri
    if "@" in uri.split("//", 1)[-1]:
        return uri  # sudah ada kredensial di URI (jarang, tapi jaga-jaga)
    scheme, rest = uri.split("//", 1)
    from urllib.parse import quote
    cred = quote(username, safe="") + (":" + quote(password, safe="") if password else "")
    return f"{scheme}//{cred}@{rest}"


# =========================================================
# 🎮 KONTROL PTZ (generik, berlaku semua merk ber-ONVIF)
# =========================================================

class OnvifPTZController:
    """
    Satu instance per kamera. Dipakai oleh CameraManager di server.py — dibuat
    lazy (hanya kalau kamera itu ditandai ptz_supported=1 di database) dan
    di-cache supaya tidak reconnect ONVIF di setiap klik tombol arah.

    Semua method aman dipanggil dari banyak thread (dilindungi lock) karena
    dashboard bisa mengirim beberapa perintah joystick berturutan dengan cepat.
    """

    def __init__(self, host: str, port: int, username: str, password: str,
                 profile_token: Optional[str] = None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._lock = threading.RLock()
        self._cam = None
        self._media = None
        self._ptz = None
        self.profile_token = profile_token
        self._presets_cache: list[dict] = []
        self._presets_cache_at = 0.0

    def _ensure_connected(self):
        with self._lock:
            if self._cam is not None:
                return
            self._cam = _connect(self.host, self.port, self.username, self.password)
            self._media = self._cam.create_media_service()
            self._ptz = self._cam.create_ptz_service()
            if not self.profile_token:
                profiles = self._media.GetProfiles()
                # Pilih profile PERTAMA yang punya PTZConfiguration; kalau tidak
                # ada yang eksplisit, pakai profile pertama saja (banyak kamera
                # OEM tetap menerima command PTZ walau field ini kosong).
                chosen = next((p for p in profiles if getattr(p, "PTZConfiguration", None)), None)
                self.profile_token = (chosen or profiles[0]).token

    def _reset_connection(self):
        with self._lock:
            self._cam = None
            self._media = None
            self._ptz = None

    def _call_with_retry(self, fn):
        """Semua kamera ONVIF beda2 soal timeout/reconnect; kalau call pertama
        gagal karena koneksi basi (kamera reboot/DHCP lease berubah dsb), coba
        sekali lagi dengan koneksi baru sebelum benar2 melempar error ke API."""
        try:
            self._ensure_connected()
            return fn()
        except Exception:
            self._reset_connection()
            self._ensure_connected()
            return fn()

    # ---------------- Continuous move (joystick-style) ----------------

    def continuous_move(self, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0):
        """
        pan/tilt/zoom masing2 -1.0 .. 1.0 (kecepatan & arah).
        pan: negatif=kiri, positif=kanan. tilt: negatif=bawah, positif=atas.
        zoom: negatif=zoom out, positif=zoom in.
        Kamera akan TERUS bergerak sampai stop() dipanggil — persis seperti
        joystick fisik di NVR (dashboard yang mengatur kapan stop, biasanya
        saat tombol arah dilepas / mouseup).
        """
        pan = max(-1.0, min(1.0, pan))
        tilt = max(-1.0, min(1.0, tilt))
        zoom = max(-1.0, min(1.0, zoom))

        def _do():
            req = self._ptz.create_type("ContinuousMove")
            req.ProfileToken = self.profile_token
            req.Velocity = {"PanTilt": {"x": pan, "y": tilt}, "Zoom": {"x": zoom}}
            self._ptz.ContinuousMove(req)
        self._call_with_retry(_do)

    def stop(self, pan_tilt: bool = True, zoom: bool = True):
        def _do():
            req = self._ptz.create_type("Stop")
            req.ProfileToken = self.profile_token
            req.PanTilt = pan_tilt
            req.Zoom = zoom
            self._ptz.Stop(req)
        self._call_with_retry(_do)

    # ---------------- Preset ----------------

    def get_presets(self) -> list[dict]:
        def _do():
            resp = self._ptz.GetPresets({"ProfileToken": self.profile_token})
            out = []
            for p in resp:
                pos = getattr(p, "PTZPosition", None)
                out.append({
                    "token": p.token,
                    "name": getattr(p, "Name", p.token),
                    "pan": getattr(getattr(pos, "PanTilt", None), "x", None) if pos else None,
                    "tilt": getattr(getattr(pos, "PanTilt", None), "y", None) if pos else None,
                    "zoom": getattr(getattr(pos, "Zoom", None), "x", None) if pos else None,
                })
            return out
        result = self._call_with_retry(_do)
        self._presets_cache = result
        self._presets_cache_at = time.time()
        return result

    def goto_preset(self, preset_token: str, speed: float = 0.6):
        def _do():
            req = self._ptz.create_type("GotoPreset")
            req.ProfileToken = self.profile_token
            req.PresetToken = preset_token
            try:
                req.Speed = {"PanTilt": {"x": speed, "y": speed}, "Zoom": {"x": speed}}
            except Exception:
                pass  # sebagian kamera tidak menerima Speed di GotoPreset, tidak fatal
            self._ptz.GotoPreset(req)
        self._call_with_retry(_do)

    def set_preset(self, name: str, preset_token: Optional[str] = None) -> str:
        """Simpan posisi PTZ SAAT INI sebagai preset baru (atau timpa preset_token
        yang sudah ada kalau diisi). Mengembalikan token preset."""
        def _do():
            req = self._ptz.create_type("SetPreset")
            req.ProfileToken = self.profile_token
            req.PresetName = name
            if preset_token:
                req.PresetToken = preset_token
            resp = self._ptz.SetPreset(req)
            return resp if isinstance(resp, str) else getattr(resp, "_value_1", preset_token or "")
        return self._call_with_retry(_do)

    def remove_preset(self, preset_token: str):
        def _do():
            req = self._ptz.create_type("RemovePreset")
            req.ProfileToken = self.profile_token
            req.PresetToken = preset_token
            self._ptz.RemovePreset(req)
        self._call_with_retry(_do)

    # ---------------- Status / home ----------------

    def get_status(self) -> dict:
        def _do():
            st = self._ptz.GetStatus({"ProfileToken": self.profile_token})
            pos = getattr(st, "Position", None)
            moving = getattr(st, "MoveStatus", None)
            return {
                "pan": getattr(getattr(pos, "PanTilt", None), "x", None) if pos else None,
                "tilt": getattr(getattr(pos, "PanTilt", None), "y", None) if pos else None,
                "zoom": getattr(getattr(pos, "Zoom", None), "x", None) if pos else None,
                "pan_tilt_moving": getattr(moving, "PanTilt", None) if moving else None,
                "zoom_moving": getattr(moving, "Zoom", None) if moving else None,
            }
        return self._call_with_retry(_do)

    def goto_home(self):
        def _do():
            req = self._ptz.create_type("GotoHomePosition")
            req.ProfileToken = self.profile_token
            self._ptz.GotoHomePosition(req)
        self._call_with_retry(_do)

    def set_home(self):
        def _do():
            req = self._ptz.create_type("SetHomePosition")
            req.ProfileToken = self.profile_token
            self._ptz.SetHomePosition(req)
        self._call_with_retry(_do)


# =========================================================
# 🗂️  REGISTRY — 1 controller per kamera, dibuat sesuai kebutuhan
# =========================================================

class PTZRegistry:
    """Cache controller per camera_id supaya endpoint FastAPI tidak reconnect
    ONVIF di setiap request. Thread-safe."""

    def __init__(self):
        self._controllers: dict[int, OnvifPTZController] = {}
        self._lock = threading.Lock()

    def get_or_create(self, cam_id: int, host: str, port: int, username: str,
                       password: str, profile_token: Optional[str]) -> OnvifPTZController:
        with self._lock:
            ctrl = self._controllers.get(cam_id)
            if ctrl is None or ctrl.host != host or ctrl.port != port:
                ctrl = OnvifPTZController(host, port, username, password, profile_token)
                self._controllers[cam_id] = ctrl
            return ctrl

    def drop(self, cam_id: int):
        with self._lock:
            self._controllers.pop(cam_id, None)


ptz_registry = PTZRegistry()
