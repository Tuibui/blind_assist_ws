#!/usr/bin/env python3
"""Generate a clean orthogonal ROS2 node diagram (with user I/O) as HTML+SVG."""
import base64

W, H = 1880, 800

# palette (light theme: white background, white boxes, coloured borders)
BG = "#ffffff"
PURPLE, PURPLE_E = "#ffffff", "#8250df"
GREEN,  GREEN_E  = "#ffffff", "#1a7f37"
BLUE,   BLUE_E   = "#ffffff", "#0969da"
GREY,   GREY_E   = "#ffffff", "#6e7681"
USER,   USER_E   = "#ffffff", "#bc6516"
YELLOW           = "#9a6700"
GREEN_T, BLUE_T, PURPLE_T, YELLOW_T, USER_T = "#1a7f37", "#0969da", "#8250df", "#9a6700", "#bc6516"

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()
ICON_IN  = b64("/home/tuibui/Downloads/voice-command.png")
ICON_OUT = b64("/home/tuibui/Downloads/listening.png")

parts = []
def add(s): parts.append(s)

# ---- nodes: (x, y, w, h) top-left based ----
USER_IN  = (55,  320, 140, 165)
SL       = (330, 330, 255, 120)
MP       = (740, 270, 245, 215)
DA       = (1255,270, 275, 215)
WD       = (1255,665, 275, 120)
USER_OUT = (1690,300, 140, 165)

def cx(n): return n[0] + n[2] / 2
def cy(n): return n[1] + n[3] / 2
def right(n): return n[0] + n[2]
def bottom(n): return n[1] + n[3]

def node(n, fill, edge, title, sub, sub2=None):
    x, y, w, h = n
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
        f'fill="{fill}" stroke="{edge}" stroke-width="2.5"/>')
    tx = x + w / 2
    if sub2:
        add(f'<text x="{tx}" y="{y+h/2-14}" class="nt">{title}</text>')
        add(f'<text x="{tx}" y="{y+h/2+12}" class="ns">{sub}</text>')
        add(f'<text x="{tx}" y="{y+h/2+34}" class="ns">{sub2}</text>')
    else:
        add(f'<text x="{tx}" y="{y+h/2-4}" class="nt">{title}</text>')
        add(f'<text x="{tx}" y="{y+h/2+20}" class="ns">{sub}</text>')

def user_node(n, icon, title, sub):
    x, y, w, h = n
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" '
        f'fill="{USER}" stroke="{USER_E}" stroke-width="2.5"/>')
    isz = 96
    add(f'<image x="{x+(w-isz)/2}" y="{y+12}" width="{isz}" height="{isz}" '
        f'href="data:image/png;base64,{icon}"/>')
    tx = x + w / 2
    add(f'<text x="{tx}" y="{y+h-32}" class="ut">{title}</text>')
    add(f'<text x="{tx}" y="{y+h-12}" class="us">{sub}</text>')

def lbl(x, y, text, color):
    add(f'<text x="{x}" y="{y}" class="el" fill="{color}">{text}</text>')

def path(d, color, dash=False, marker=True):
    da = ' stroke-dasharray="7 6"' if dash else ''
    mk = ' marker-end="url(#arrow)"' if marker else ''
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.6"{da}{mk} '
        f'stroke-linejoin="round" stroke-linecap="round"/>')

# ================= EDGES (under nodes) =================

# ---- USER speaks -> speech_logger ----
yin = cy(SL)
path(f"M {right(USER_IN)} {yin} H {SL[0]}", USER_E)
lbl((right(USER_IN) + SL[0]) / 2, yin - 10, "voice command", USER_T)

# ---- MP -> DA : 5 straight horizontal lines ----
det = [
    ("current mode",       YELLOW, YELLOW_T),
    ("banknote result",    GREEN_E, GREEN_T),
    ("obstacle (L/C/R)",   GREEN_E, GREEN_T),
    ("recognized face",    GREEN_E, GREEN_T),
    ("walking path",       GREEN_E, GREEN_T),
]
ys = [300, 337, 374, 411, 448]
xL, xR = right(MP), DA[0]
for (text, col, tcol), yy in zip(det, ys):
    path(f"M {xL} {yy} H {xR}", col)
    lbl((xL + xR) / 2, yy - 8, text, tcol)

# ---- SL -> MP : two commands ----
xs, xm = right(SL), MP[0]
for yy, text in [(355, "switch mode"), (400, "navigation on/off")]:
    path(f"M {xs} {yy} H {xm}", PURPLE_E)
    lbl((xs + xm) / 2, yy - 8, text, PURPLE_T)

# ---- MP -> SL : current mode (loop back, clearly under MP) ----
yloop = 560
path(f"M {MP[0]+45} {bottom(MP)} V {yloop} H {cx(SL)} V {bottom(SL)}", YELLOW)
lbl((MP[0] + 45 + cx(SL)) / 2, yloop - 8, "current mode", YELLOW_T)

# ---- MP -> WD : live video + current mode ----
path(f"M {cx(MP)} {bottom(MP)} V 610 H {cx(WD)-40} V {WD[1]}", GREEN_E)
lbl(cx(MP) + 110, 602, "live video", GREEN_T)
y_cm = cy(WD)
path(f"M {cx(MP)+55} {bottom(MP)} V {y_cm} H {WD[0]}", YELLOW)
lbl((cx(MP) + 55 + WD[0]) / 2, y_cm - 8, "current mode", YELLOW_T)

# ---- DA -> WD : spoken text (vertical) ----
path(f"M {cx(DA)} {bottom(DA)} V {WD[1]}", BLUE_E)
lbl(cx(DA) + 70, (bottom(DA) + WD[1]) / 2, "spoken text", BLUE_T)

# ---- DA -> USER hears : spoken result ----
yout = cy(USER_OUT)
path(f"M {right(DA)} {cy(DA)} V {yout} H {USER_OUT[0]}", BLUE_E)
lbl((right(DA) + USER_OUT[0]) / 2, yout - 10, "spoken result", BLUE_T)

# ================= NODES (on top) =================
user_node(USER_IN, ICON_IN, "User", "speaks a command")
node(SL, PURPLE, PURPLE_E, "speech_logger_node", "listens for voice commands")
node(MP, GREEN, GREEN_E, "main_pipeline_node", "camera + AI", "processing")
node(DA, BLUE, BLUE_E, "decision_audio_node", "decides +", "speaks out loud")
node(WD, BLUE, BLUE_E, "web_display_node", "web dashboard (:8080)")
user_node(USER_OUT, ICON_OUT, "User", "hears the result")

svg_body = "\n".join(parts)

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body {{ margin:0; background:{BG}; }}
  svg {{ display:block; }}
  text {{ font-family:'Helvetica','Arial',sans-serif; }}
  .title {{ font-size:31px; font-weight:800; fill:#1f2328; text-anchor:middle; }}
  .subtitle {{ font-size:15px; fill:#656d76; text-anchor:middle; }}
  .nt {{ font-size:18px; font-weight:800; fill:#1f2328; text-anchor:middle; }}
  .ns {{ font-size:13px; fill:#57606a; text-anchor:middle; }}
  .ut {{ font-size:17px; font-weight:800; fill:#bc6516; text-anchor:middle; }}
  .us {{ font-size:12px; fill:#8a6a44; text-anchor:middle; }}
  .el {{ font-size:14px; text-anchor:middle; font-weight:700; }}
</style></head><body>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrow" markerWidth="14" markerHeight="14" refX="10" refY="5"
            orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L11,5 L0,10 z" fill="#57606a"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="{BG}"/>
  <text x="{W/2}" y="50" class="title">System Diagram</text>
  <g transform="translate(0,-160)">
  {svg_body}
  </g>
</svg></body></html>"""

with open("node_diagram.html", "w") as f:
    f.write(html)
print("wrote node_diagram.html")
