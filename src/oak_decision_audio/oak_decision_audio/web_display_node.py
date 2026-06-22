"""Single-page local web dashboard for the blind-assist system.

Shows four things:
  1. a live camera video stream (MJPEG)
  2. the current mode (Walk / Money)
  3. the latest alert/announcement text (what the user is being told)
  4. the LiPo battery voltage / percent (read from an ADS1115 on A0)

No OpenCV window. Open http://<pi-ip>:8080 in a browser.

Battery wiring (this build):
  3S LiPo (+) --[ 1.2 kOhm ]--+--[ 0.67 kOhm ]-- GND
                              |
                         ADS1115 A0
The ADC reads the voltage across the 0.67 kOhm resistor. Because a full 3S pack
is 12.6 V, the divided voltage reaches ~4.51 V, so the ADS1115 must be powered
from 5 V (VDD) and use the +/-6.144 V gain range. See the battery_* parameters.
"""

import fcntl
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, String

from oak_interfaces.msg import Mode


# --- Minimal, dependency-free ADS1115 reader over /dev/i2c-N --------------
# We talk to the chip directly with ioctl/read/write so the node needs no
# extra Python packages (no adafruit-blinka / smbus2). Single-shot reads.

_I2C_SLAVE = 0x0703  # ioctl request to set the I2C target address
_ADS_REG_CONVERT = 0x00
_ADS_REG_CONFIG = 0x01

# Full-scale range (volts) -> PGA bits for the config register.
_ADS_FSR_TO_PGA = {
    6.144: 0b000,
    4.096: 0b001,
    2.048: 0b010,
    1.024: 0b011,
    0.512: 0b100,
    0.256: 0b101,
}


class ADS1115:
    """Read one single-ended channel (A0..A3) of an ADS1115 in single-shot mode."""

    def __init__(self, bus: int = 1, address: int = 0x48, fsr_volts: float = 6.144):
        self._path = f"/dev/i2c-{bus}"
        self._address = address
        self._fsr = float(fsr_volts)
        self._pga = _ADS_FSR_TO_PGA.get(self._fsr, 0b000)
        self._fd = os.open(self._path, os.O_RDWR)
        fcntl.ioctl(self._fd, _I2C_SLAVE, self._address)

    def read_volts(self, channel: int = 0) -> float:
        """Return the voltage seen at the given input pin (relative to GND)."""
        mux = 0b100 | (channel & 0b011)  # 100..111 = AINx vs GND
        config = (
            (1 << 15)          # OS: start a single conversion
            | (mux << 12)      # input multiplexer
            | (self._pga << 9) # programmable gain (full-scale range)
            | (1 << 8)         # MODE: single-shot
            | (0b100 << 5)     # data rate: 128 SPS
            | 0b00011          # comparator disabled
        )
        os.write(self._fd, bytes([_ADS_REG_CONFIG, (config >> 8) & 0xFF, config & 0xFF]))
        time.sleep(0.012)  # 128 SPS conversion takes ~8 ms; allow margin
        os.write(self._fd, bytes([_ADS_REG_CONVERT]))
        raw = int.from_bytes(os.read(self._fd, 2), "big", signed=True)
        return raw * self._fsr / 32768.0

    def close(self) -> None:
        try:
            os.close(self._fd)
        except Exception:
            pass


# Per-cell voltage -> state of charge (%). Simple linear mapping between a
# "full" and "empty" cell voltage. For a 3S pack this gives:
#   4.20 V/cell -> 12.6 V = 100%,  3.50 V/cell -> 10.5 V = 0% (safety margin).
_LIPO_FULL_CELL = 4.20   # V per cell at 100% (12.6 V pack)
_LIPO_EMPTY_CELL = 3.50  # V per cell at 0%  (10.5 V pack, safety cutoff)


def _lipo_percent(cell_volts: float) -> int:
    frac = (cell_volts - _LIPO_EMPTY_CELL) / (_LIPO_FULL_CELL - _LIPO_EMPTY_CELL)
    return int(round(max(0.0, min(1.0, frac)) * 100))


PAGE = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind Assist</title>
<style>
  :root { --bg:#f4f5f7; --card:#ffffff; --line:#e4e7eb; --text:#1f2329;
    --muted:#8a9199; --walk:#1f9d57; --money:#e08a00; --face:#2f7bd6; --red:#d64545; }
  * { box-sizing:border-box; }
  html, body { margin:0; min-height:100%; background:var(--bg); color:var(--text);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Thai",sans-serif;
    -webkit-font-smoothing:antialiased; }
  .wrap { max-width:760px; margin:0 auto; padding:20px 16px 28px; }
  header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:18px; }
  header h1 { font-size:18px; font-weight:600; margin:0; }
  .link { font-size:13px; color:var(--muted); }
  .link b { color:var(--walk); font-weight:600; }
  .link b.off { color:var(--red); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
    overflow:hidden; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.05); }
  .label { font-size:12px; color:var(--muted); letter-spacing:.06em; text-transform:uppercase;
    padding:13px 16px 0; }
  .feed { background:#000; aspect-ratio:16/9; display:flex; align-items:center;
    justify-content:center; margin-top:10px; }
  .feed img { width:100%; height:100%; object-fit:contain; }
  .pad { padding:6px 16px 18px; }
  .mode { font-size:clamp(30px,8vw,56px); font-weight:700; line-height:1.1; }
  .mode.walk { color:var(--walk); } .mode.money { color:var(--money); } .mode.face { color:var(--face); }
  .alert { font-size:clamp(22px,5vw,40px); font-weight:600; min-height:1.3em; word-break:break-word; }
  .alert.stale { color:var(--muted); }
  .batt { font-size:clamp(28px,7vw,48px); font-weight:700; line-height:1.1; }
  .batt.ok { color:var(--walk); } .batt.low { color:var(--money); } .batt.crit { color:var(--red); }
  .batt.stale { color:var(--muted); }
  .batt-pct { font-size:clamp(16px,4vw,24px); font-weight:600; color:var(--muted); margin-left:10px; }
  footer { display:flex; gap:20px; flex-wrap:wrap; font-size:12px; color:var(--muted); padding:0 4px; }
  footer b { color:var(--text); font-weight:600; }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Blind Assist</h1>
      <span class="link">เชื่อมต่อ <b id="conn" class="off">—</b></span>
    </header>

    <div class="card">
      <div class="label">กล้อง / Camera</div>
      <div class="feed"><img id="cam" alt="ไม่มีสัญญาณ"></div>
    </div>

    <div class="card">
      <div class="label">โหมด / Mode</div>
      <div class="pad"><span id="mode" class="mode">—</span></div>
    </div>

    <div class="card">
      <div class="label">แบตเตอรี่ / Battery</div>
      <div class="pad">
        <span id="batt" class="batt">—</span>
        <span id="battpct" class="batt-pct">—</span>
      </div>
    </div>

    <div class="card">
      <div class="label">แจ้งเตือน / Alert</div>
      <div class="pad"><span id="alert" class="alert">—</span></div>
    </div>

    <footer>
      <span>เวลา <b id="clock">--:--:--</b></span>
      <span>อัปเดตล่าสุด <b id="age">—</b> วิ</span>
    </footer>
  </div>
<script>
// MJPEG stream; reload on error so it recovers if the publisher restarts.
const cam = document.getElementById('cam');
function startStream() { cam.src = '/stream?ts=' + Date.now(); }
cam.onerror = () => setTimeout(startStream, 1500);
startStream();

function pad(n){ return String(n).padStart(2,'0'); }
setInterval(() => {
  const d = new Date();
  document.getElementById('clock').textContent =
    pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());
}, 1000);

async function tick() {
  try {
    const r = await fetch('/state', {cache: 'no-store'});
    const s = await r.json();
    const mode = document.getElementById('mode');
    mode.textContent = s.mode_text || '—';
    mode.className = 'mode ' + (s.mode || '');
    const alert = document.getElementById('alert');
    alert.textContent = s.announcement || '—';
    alert.className = 'alert' + ((s.age_sec > 8) ? ' stale' : '');
    const batt = document.getElementById('batt');
    const battpct = document.getElementById('battpct');
    if (s.batt_volts == null) {
      batt.textContent = '—'; batt.className = 'batt stale'; battpct.textContent = '';
    } else {
      batt.textContent = s.batt_volts.toFixed(2) + ' V';
      battpct.textContent = (s.batt_percent != null ? s.batt_percent + ' %' : '');
      let cls = (s.batt_age > 30) ? 'stale'
              : (s.batt_percent != null && s.batt_percent <= 15) ? 'crit'
              : (s.batt_percent != null && s.batt_percent <= 35) ? 'low' : 'ok';
      batt.className = 'batt ' + cls;
    }
    document.getElementById('age').textContent =
      (s.age_sec >= 9999) ? '—' : s.age_sec;
    const conn = document.getElementById('conn');
    conn.textContent = 'ปกติ'; conn.className = '';
  } catch (e) {
    const conn = document.getElementById('conn');
    conn.textContent = 'ขาด'; conn.className = 'off';
  }
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>"""


class WebDisplayNode(Node):
    def __init__(self) -> None:
        super().__init__("web_display_node")

        self.declare_parameter("current_mode_topic", "/oak/current_mode")
        self.declare_parameter("announcement_topic", "/oak/announcement")
        self.declare_parameter("preview_topic", "/oak/preview/compressed")
        self.declare_parameter("http_host", "0.0.0.0")
        self.declare_parameter("http_port", 8080)

        # Battery monitor (ADS1115 on A0 reading a 1.2k / 0.67k divider).
        self.declare_parameter("battery_enable", True)
        self.declare_parameter("battery_i2c_bus", 1)
        self.declare_parameter("battery_i2c_addr", 0x48)
        self.declare_parameter("battery_channel", 0)        # A0
        self.declare_parameter("battery_fsr_volts", 6.144)  # +/-6.144 V range
        self.declare_parameter("battery_r_top_ohms", 1200.0)
        self.declare_parameter("battery_r_bottom_ohms", 670.0)
        self.declare_parameter("battery_calibration", 1.0)  # fine-tune multiplier
        self.declare_parameter("battery_cells", 3)          # 3S LiPo
        self.declare_parameter("battery_poll_sec", 5.0)
        # เผยแพร่ % แบตให้ decision_audio_node เอาไปพูดเตือนเมื่อแบตต่ำ
        self.declare_parameter("battery_percent_topic", "/oak/battery_percent")

        mode_topic = str(self.get_parameter("current_mode_topic").value)
        announcement_topic = str(self.get_parameter("announcement_topic").value)
        preview_topic = str(self.get_parameter("preview_topic").value)
        host = str(self.get_parameter("http_host").value)
        port = int(self.get_parameter("http_port").value)

        self._lock = threading.Lock()
        self._mode = ""          # "walk" | "money" | ""
        self._mode_text = "—"
        self._announcement = ""
        self._announcement_ts = 0.0
        self._batt_volts = None   # pack voltage, or None until first good read
        self._batt_percent = None
        self._batt_ts = 0.0

        # Latest JPEG frame for the MJPEG stream, guarded by a Condition so
        # stream threads block until a new frame arrives (no busy polling).
        self._frame_cond = threading.Condition()
        self._frame = b""
        self._frame_seq = 0

        self.create_subscription(Mode, mode_topic, self._on_mode, 10)
        self.create_subscription(String, announcement_topic, self._on_announcement, 10)
        self.create_subscription(CompressedImage, preview_topic, self._on_preview, 10)

        # เผยแพร่ % แบต (อ่านได้ในเธรดแบต) ให้ node เสียงเอาไปเตือนผู้ใช้
        self._batt_pct_pub = self.create_publisher(
            Int32, str(self.get_parameter("battery_percent_topic").value), 10
        )

        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        shown_host = host if host not in ("0.0.0.0", "") else "<this-device-ip>"
        self.get_logger().info(f"Web dashboard at http://{shown_host}:{port} (and http://localhost:{port})")

        # Battery polling thread (best-effort; the dashboard works without it).
        self._batt_stop = threading.Event()
        self._batt_thread = None
        if bool(self.get_parameter("battery_enable").value):
            self._batt_thread = threading.Thread(target=self._battery_loop, daemon=True)
            self._batt_thread.start()

    def _battery_loop(self) -> None:
        bus = int(self.get_parameter("battery_i2c_bus").value)
        addr = int(self.get_parameter("battery_i2c_addr").value)
        channel = int(self.get_parameter("battery_channel").value)
        fsr = float(self.get_parameter("battery_fsr_volts").value)
        r_top = float(self.get_parameter("battery_r_top_ohms").value)
        r_bottom = float(self.get_parameter("battery_r_bottom_ohms").value)
        cal = float(self.get_parameter("battery_calibration").value)
        cells = max(1, int(self.get_parameter("battery_cells").value))
        period = max(1.0, float(self.get_parameter("battery_poll_sec").value))
        scale = (r_top + r_bottom) / r_bottom * cal  # pin volts -> pack volts

        adc = None
        warned = False
        while not self._batt_stop.is_set():
            try:
                if adc is None:
                    adc = ADS1115(bus=bus, address=addr, fsr_volts=fsr)
                    self.get_logger().info(
                        f"Battery monitor: ADS1115 on /dev/i2c-{bus} addr 0x{addr:02x}, "
                        f"A{channel}, divider x{scale:.3f}"
                    )
                    warned = False
                pin_v = adc.read_volts(channel)
                pack_v = pin_v * scale
                percent = _lipo_percent(pack_v / cells)
                with self._lock:
                    self._batt_volts = pack_v
                    self._batt_percent = percent
                    self._batt_ts = time.monotonic()
                # บอก % แบตล่าสุดให้ decision_audio_node (ใช้ตัดสินใจเตือนแบตต่ำ)
                self._batt_pct_pub.publish(Int32(data=int(percent)))
            except Exception as exc:  # noqa: BLE001 - hardware can fail any time
                if adc is not None:
                    adc.close()
                adc = None
                with self._lock:
                    self._batt_volts = None
                    self._batt_percent = None
                if not warned:
                    self.get_logger().warning(f"Battery read failed: {exc}")
                    warned = True
            self._batt_stop.wait(period)
        if adc is not None:
            adc.close()

    def _on_mode(self, msg: Mode) -> None:
        with self._lock:
            if msg.mode == Mode.MONEY:
                self._mode, self._mode_text = "money", "โหมดเงิน / Money"
            elif msg.mode == Mode.FACE:
                self._mode, self._mode_text = "face", "โหมดเพื่อน / Face"
            else:
                self._mode, self._mode_text = "walk", "โหมดเดิน / Walk"

    def _on_announcement(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        with self._lock:
            self._announcement = text
            self._announcement_ts = time.monotonic()

    def _on_preview(self, msg: CompressedImage) -> None:
        data = bytes(msg.data)
        if not data:
            return
        with self._frame_cond:
            self._frame = data
            self._frame_seq += 1
            self._frame_cond.notify_all()

    def _state_json(self) -> bytes:
        with self._lock:
            age = time.monotonic() - self._announcement_ts if self._announcement_ts else 9999.0
            batt_age = (time.monotonic() - self._batt_ts) if self._batt_ts else 9999.0
            payload = {
                "mode": self._mode,
                "mode_text": self._mode_text,
                "announcement": self._announcement,
                "age_sec": round(age, 1),
                "batt_volts": round(self._batt_volts, 2) if self._batt_volts is not None else None,
                "batt_percent": self._batt_percent,
                "batt_age": round(batt_age, 1),
            }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _make_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence per-request stderr logging
                pass

            def _send(self, code, body, content_type):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/state"):
                    self._send(200, node._state_json(), "application/json; charset=utf-8")
                elif self.path.startswith("/stream"):
                    self._stream()
                elif self.path in ("/", "/index.html"):
                    self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain; charset=utf-8")

            def _stream(self):
                try:
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    return
                last_seq = -1
                while True:
                    with node._frame_cond:
                        if node._frame_seq == last_seq:
                            node._frame_cond.wait(timeout=2.0)
                        frame = node._frame
                        seq = node._frame_seq
                    if seq == last_seq or not frame:
                        continue
                    last_seq = seq
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(frame)).encode("ascii")
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

        return Handler

    def shutdown(self) -> None:
        self._batt_stop.set()
        try:
            self._server.shutdown()
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
