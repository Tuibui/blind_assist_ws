"""Single-page local web dashboard for the blind-assist system.

Shows three things:
  1. a live camera video stream (MJPEG)
  2. the current mode (Walk / Money)
  3. the latest alert/announcement text (what the user is being told)

No OpenCV window. Open http://<pi-ip>:8080 in a browser.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from oak_interfaces.msg import Mode


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

        # Latest JPEG frame for the MJPEG stream, guarded by a Condition so
        # stream threads block until a new frame arrives (no busy polling).
        self._frame_cond = threading.Condition()
        self._frame = b""
        self._frame_seq = 0

        self.create_subscription(Mode, mode_topic, self._on_mode, 10)
        self.create_subscription(String, announcement_topic, self._on_announcement, 10)
        self.create_subscription(CompressedImage, preview_topic, self._on_preview, 10)

        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        shown_host = host if host not in ("0.0.0.0", "") else "<this-device-ip>"
        self.get_logger().info(f"Web dashboard at http://{shown_host}:{port} (and http://localhost:{port})")

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
            payload = {
                "mode": self._mode,
                "mode_text": self._mode_text,
                "announcement": self._announcement,
                "age_sec": round(age, 1),
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
