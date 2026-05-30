"""ทำให้ระบบเสียงของผู้ใช้ "ต่อถึง" ได้ ไม่ว่าจะสั่งรัน node ด้วยวิธีไหน

บน Raspberry Pi ทั้งฝั่งเล่นเสียง (edge-playback/mpv/ffplay) และฝั่งอัดเสียง
(arecord และปลั๊กอิน ALSA->PipeWire ที่ pyaudio ใช้) จะคุยกับ audio server ได้
ก็ต่อเมื่อ environment variable เหล่านี้ชี้ไปที่ session ของ PipeWire/PulseAudio
ที่กำลังทำงานอยู่:

    XDG_RUNTIME_DIR           เช่น /run/user/1000
    PULSE_SERVER              unix:/run/user/1000/pulse/native
    DBUS_SESSION_BUS_ADDRESS  unix:path=/run/user/1000/bus

ถ้าล็อกอินหน้าจอ (desktop) ตัวแปรพวกนี้จะมีให้อยู่แล้ว แต่ถ้าสั่งผ่าน SSH จาก
เครื่องอื่น "จะไม่มี" ทำให้เสียงเงียบหายไปเฉย ๆ ฟังก์ชันนี้จึงเติมค่าที่ขาดให้
(โดยไม่ทับค่าที่ผู้ใช้ตั้งเองมาแล้ว) อ้างอิงจาก session /run/user/<uid> ของผู้ใช้
ปัจจุบัน และเพราะเซ็ตลงใน os.environ ทั้ง subprocess และ ALSA ในโปรเซสจึงได้รับค่านี้ไปด้วย
"""

import os


def ensure_audio_session_env():
    """เติมตัวแปร environment ของ audio session ที่ขาดไป; คืน list ว่าเซ็ตอะไรไปบ้าง"""
    applied = []

    # 1) XDG_RUNTIME_DIR — ถ้าไม่มีหรือชี้ไปโฟลเดอร์ที่ไม่มีจริง ให้เดาเป็น /run/user/<uid>
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime_dir or not os.path.isdir(runtime_dir):
        candidate = f"/run/user/{os.getuid()}"
        if os.path.isdir(candidate):
            os.environ["XDG_RUNTIME_DIR"] = candidate
            runtime_dir = candidate
            applied.append(f"XDG_RUNTIME_DIR={candidate}")

    # หา runtime dir ไม่ได้เลย ก็ทำอะไรต่อไม่ได้
    if not runtime_dir:
        return applied

    # 2) PULSE_SERVER — ชี้ไป socket ของ PulseAudio/PipeWire ใน runtime dir
    if not os.environ.get("PULSE_SERVER"):
        pulse_sock = os.path.join(runtime_dir, "pulse", "native")
        if os.path.exists(pulse_sock):
            value = f"unix:{pulse_sock}"
            os.environ["PULSE_SERVER"] = value
            applied.append(f"PULSE_SERVER={value}")

    # 3) DBUS_SESSION_BUS_ADDRESS — ชี้ไป D-Bus session bus ใน runtime dir
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        bus_sock = os.path.join(runtime_dir, "bus")
        if os.path.exists(bus_sock):
            value = f"unix:path={bus_sock}"
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = value
            applied.append(f"DBUS_SESSION_BUS_ADDRESS={value}")

    return applied
