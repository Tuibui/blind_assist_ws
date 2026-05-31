# Blind Assist Workspace — Handoff

ROS2 workspace on Raspberry Pi + OAK-D Lite that helps a blind user:
- **Walk mode** (boot default): obstacle detection + spatial XYZ — alerts when something is in front and close
- **Money mode**: classify THB banknotes via DepthAI NN Archive, speak result
- **Voice control**: speech_logger listens, switches modes and language
- **Language**: TH default, runtime switch TH↔EN, single-language at a time (no bilingual)

## Packages

```
src/
├── oak_bringup/           launch + system config (ament_cmake)
├── oak_interfaces/        msgs: Mode, Detection2D, Detection2DArray
├── oak_detectors/         main_pipeline (walk + money in one node)
│   ├── oak_detectors/     runtime: main_pipeline.py + tractile_model/scripts/color_path_detect.py
│   ├── models/            money/best.rvc2.tar.xz (walk is depth-only, no model)
│   └── tools/             offline helpers + reference pipelines (NOT installed at runtime)
└── oak_decision_audio/    decision_audio_node, speech_logger_node
```

The 4 packages above are all that exist. Only 3 nodes run (see launch):
`oak_detectors/main_pipeline`, `oak_decision_audio/{decision_audio_node,speech_logger_node,web_display_node}`.

## Topics

| Topic | Type | Pub | Sub |
|---|---|---|---|
| `/oak/mode_cmd` | `oak_interfaces/Mode` | speech_logger | main_pipeline |
| `/oak/current_mode` | `oak_interfaces/Mode` | main_pipeline | decision_audio, web_display |
| `/oak/language` | `std_msgs/String` ("th"/"en") | (none — manual override) | both audio nodes |
| `/money/detections` | `Detection2DArray` | main_pipeline | decision_audio |
| `/navigation/detections` | `Detection2DArray` | main_pipeline | decision_audio |
| `/face/detections` | `Detection2DArray` | main_pipeline | decision_audio |
| `/oak/announcement` | `std_msgs/String` | decision_audio | web_display |
| `/oak/preview/compressed` | `sensor_msgs/CompressedImage` (JPEG) | main_pipeline | web_display |

Mode enum: `Mode.WALK=0`, `Mode.MONEY=1`, `Mode.FACE=2`. Default boot mode = WALK.

## Language — set at launch (no longer by voice)

Language is chosen once via a launch argument; voice-command language switching was removed.

```bash
ros2 launch oak_bringup blind_assist_system.launch.py language:=en   # english
ros2 launch oak_bringup blind_assist_system.launch.py language:=th   # thai (default)
```

Accepts `en`/`english` or `th`/`thai`. The launch arg overrides `default_language` on
`decision_audio_node` + `speech_logger_node`. The `/oak/language` topic still exists as an
optional runtime override (`ros2 topic pub /oak/language std_msgs/String "data: en"`), but
nothing publishes to it automatically anymore.

## Face mode (mode 3) — friend recognition

Say **"สวัสดี" / "ดีครับ" / "ดีค่ะ"** (or "hello"/"hi") → speech_logger publishes `Mode.FACE`.
main_pipeline runs the `myfriend_face` NN Archive (same DetectionNetwork builder as money,
classes = `["Tee", "Ping"]`) and publishes detections on `/face/detections`. decision_audio
**speaks the detected person's name directly** (e.g. "Tee") and shows it on the web dashboard.
If nobody is detected it stays silent. The mode switch itself is **not announced** —
`decision_audio._on_mode` only announces WALK/MONEY, so FACE is silent by design. The same
name is not repeated within `face_repeat_sec` (default 5 s); a different name speaks at once.

## Web dashboard (no GUI windows)

There is **no OpenCV/GUI window** anywhere at runtime. `web_display_node` serves a
single page at **http://<device-ip>:8080** (also `http://localhost:8080`) showing:
1. a **live video stream** (MJPEG at `/stream`) — from `/oak/preview/compressed`
2. current **mode** (Walk / Money) — from `/oak/current_mode`
3. the latest **alert text** (what the user is being told) — from `/oak/announcement`

The page polls `/state` (JSON) every 500 ms and embeds `/stream` as an `<img>`; no extra
Python deps (stdlib `http.server`). `decision_audio_node` publishes `/oak/announcement`
with the exact phrase it speaks (boot greeting, mode change, money result, obstacle alert)
in the active language. `main_pipeline` publishes JPEG preview frames on
`/oak/preview/compressed` — in **walk mode** it reuses the RGB tactile camera frame (no
extra camera output), in **money mode** it requests a small 640×360 camera output. Frames
are downscaled to `preview_max_width` and JPEG-encoded at `preview_jpeg_quality`. Disable
with `preview_enabled: false` in `main_pipeline.yaml` to save CPU/USB bandwidth. Port/host
and preview topic configurable via params in `oak_decision_audio.yaml`.

## Architecture decision: ONE pipeline node, not two

`main_pipeline.py` owns both walk and money pipelines. OAK-D Lite is single-device anyway, so they're mutually exclusive. On mode switch, `_cleanup_pipeline()` closes the current pipeline and builds the new one (~1–2s switch). Tracks active branch in `self._active_mode_pipeline = "walk" | "money" | None`.

## Money pipeline details

- Model: **NN Archive (.tar.xz)** at `src/oak_detectors/models/money/best.rvc2.tar.xz` (loaded via `dai.NNArchive` + `DetectionNetwork.build`). Old `.blob` was deleted.
- Labels (model index order): `["20", "50", "100", "500", "1000"]` — no "not_found" class anymore.
- Per-class confidence thresholds: `confidence_thresholds: [0.20, 0.20, 0.40, 0.20, 0.20]` (host-side filter after on-device NMS).
- Camera: BGR888p preview, NN input from archive metadata, 30 fps, **rotated 180°**.
- `[det]` log throttled to **1 Hz** via `throttle_duration_sec=1.0` (was firing per frame).
- Reference standalone (for accuracy comparison): `src/oak_detectors/oak_detectors/money_detect_ref_pipeline.py`.

### Decision audio confirmation — now COUNT-based, not time-based

Changed from "same label for 3 sec" to **"same label 10 consecutive detection frames"**. Why: detection rate varies with conditions; counting frames is more deterministic.
- Param: `money_confirm_count: 10` (in `oak_decision_audio.yaml`). Old `money_confirm_sec` removed.
- Reset counter on label change, no detection, or during `money_gap_sec` cooldown.

## Walk pipeline details (depth-only — no NN)

**Goal**: tell the user if something is close in front and *roughly where* (left/center/right). Catches **any** obstacle (wall, pole, glass, box), not just known object classes.

- **No neural network.** The MobileNet-SSD approach was dropped: it only detected 20 VOC classes (missed walls/poles/etc.) and ate RVC2 memory.
- Stereo: CAM_B + CAM_C → `StereoDepth`, depth-aligned to CAM_A, `setOutputSize(640, 400)`, `HIGH_DENSITY` preset.
- CAM_A is still built — used only for the **tactile color path** and as the depth-alignment reference. Locked to a fixed lens (`walk_manual_focus`, default 130) so autofocus does not hunt.
- Host-side analysis (`_analyze_depth_zones`): take a vertical ROI (`walk_roi_top_frac`..`walk_roi_bottom_frac`, default 0.1–0.9 to trim ceiling/floor), split into **left/center/right** thirds. Per zone: count pixels with `0 < z <= walk_obstacle_z_max_mm`; near distance = 20th-percentile of those pixels (robust to noise).

### Obstacle alert rule (per zone)

```
near_pixel_fraction >= walk_obstacle_min_pixel_frac   (default 0.04 = 4%)
AND
0 < z_near_mm <= walk_obstacle_z_max_mm               (default 1500 mm = 1.5 m)
```

When true: log `[obstacle] <zone> <dist>m (<frac>%)` warning at 1 Hz, publish one `Detection2D` per zone to `/navigation/detections` (`class_name = "left"|"center"|"right"`, `confidence = near fraction`, `center_y = z_near_m`, bbox = zone column in normalized coords). `decision_audio` speaks the `class_name`.

### Models that did NOT work / were dropped

- `unet-camvid-onnx-0001` — 12-class CamVid seg. **OOM on RVC2** (needs 118 MB, only 103 MB available). Blob removed.
- `road-segmentation-adas-0001` — 4-class (bg/road/curb/lane). Worked memory-wise but user pivoted away from segmentation entirely to "detect + depth + alert" approach. Blob removed.
- `SpatialDetectionNetwork` — hung after first frame in DepthAI v3 (probably queue/timing issue with our setup). Switched to plain `NeuralNetwork` + manual depth lookup.

## Voice commands

| Phrase | Action |
|---|---|
| "เดินต่อไป" / EN keyword: `walk`, `forward` | → WALK mode |
| "เช็คธนบัตร" / "เช็กธนบัตร" / EN keyword: `money`, `banknote`, `cash` | → MONEY mode |
| "สวัสดี" / "ดีครับ" / "ดีค่ะ" / EN keyword: `hello`, `hi`, `hey` | → FACE mode |
| "เปิด" / EN keyword: `on`, `enable`, `start`, `resume` | → navigation ON (WALK only) |
| "ปิด" / EN keyword: `off`, `disable`, `stop`, `pause` | → navigation OFF (WALK only) |

**Partial detection — only one keyword is needed, not the full phrase.**
- **Thai** is matched as a **substring** anywhere in the sentence (Thai words run together),
  e.g. "เปิดระบบนำทางหน่อย" matches "เปิด". "เปิด" (on) is checked before "ปิด" (off)
  because "ปิด" is a substring of "เปิด".
- **English** is matched on **whole-word tokens** (`re.findall(r"[a-z]+", ...)`), so saying
  just "money" or "off" works. Token (not substring) matching is required because short
  words like "on" are substrings of other words ("navigati**on**"). Full English phrases
  ("navigation off", "check banknote", …) still match too.
- Navigation on/off is a **sub-toggle of WALK mode** (tactile path only — it does NOT change
  mode; obstacle detection keeps running). It is **gated to WALK mode**; ignored in money/face.

SR language is fixed at launch (`th-TH` / `en-US`); language is no longer switchable by voice.

## Critical conventions

### Bash wrappers, NOT entry_points

Every Python ROS2 package exposes its node via `scripts/<node_name>` + `scripts=[...]` in `setup.py`. Don't switch to `console_scripts` — bakes a hardcoded shebang to whatever python colcon saw, breaks when venv activation order is off.

```bash
#!/usr/bin/env bash
exec python3 -c 'import sys; from <pkg>.<module> import main; main(sys.argv[1:])' "$@"
```

### venv activation order matters

ROS first, then venv:
```bash
source /opt/ros/jazzy/setup.bash
source venv/bin/activate
which python3   # must be /home/abcd/blind_assist_ws/venv/bin/python3
```

If returns `/usr/bin/python3`, venv lost — rebuild won't help.

### Sourcing install/setup.bash IS required for launch

Symptom: `ModuleNotFoundError: No module named 'oak_detectors'` when launching even though other nodes start. Workspace not sourced in that shell. Always:
```bash
source install/setup.bash
ros2 launch oak_bringup blind_assist_system.launch.py
```

## Build & run

```bash
cd ~/blind_assist_ws
source /opt/ros/jazzy/setup.bash
source venv/bin/activate
colcon build --symlink-install
source install/setup.bash
ros2 launch oak_bringup blind_assist_system.launch.py
```

Standalone:
```bash
ros2 run oak_detectors main_pipeline --ros-args -p default_mode:=money
ros2 run oak_detectors main_pipeline --ros-args -p default_mode:=walk
```

Mode poke:
```bash
ros2 topic pub --once /oak/mode_cmd oak_interfaces/Mode "mode: 1"   # MONEY
ros2 topic pub --once /oak/mode_cmd oak_interfaces/Mode "mode: 0"   # WALK
ros2 topic pub --once /oak/language std_msgs/String "data: en"
```

## Key configs

`src/oak_detectors/config/main_pipeline.yaml`:
- `default_mode: walk`
- `confidence_threshold: 0.20`, `confidence_thresholds: [0.20, 0.20, 0.40, 0.20, 0.20]` (per-class money)
- `camera_fps: 30.0`, `rotate_preview_180: true`, `display_enabled: true`
- `log_window_frames: 2` (param exists, currently unused in pipeline — was for an earlier majority-vote experiment)
- Walk: `walk_input_width: 300`, `walk_input_height: 300`
- Walk: `walk_confidence_threshold: 0.5`
- Walk: `walk_obstacle_z_max_mm: 1500.0`, `walk_obstacle_x_abs_mm: 600.0`

`src/oak_decision_audio/config/oak_decision_audio.yaml`:
- `money_confirm_count: 10` (replaces old `money_confirm_sec`)
- `money_gap_sec: 3.0`
- `default_language: th`, `tts_enabled: true`

## Open issues / next session

1. **Walk depth-only — needs on-device tuning (NOT re-tested).** Walk mode was rewritten to depth-only obstacle zones (no NN). Rebuild + run on the Pi and confirm: (a) lens no longer hunts (manual focus locked), (b) `[obstacle] left/center/right` logs fire when something is within ~1.5 m, (c) no false alerts from floor/ceiling — tune `walk_roi_top_frac`/`walk_roi_bottom_frac` and `walk_obstacle_min_pixel_frac` if needed. Note: stereo struggles with blank walls / glass / low light.
2. **Money model accuracy** — user will retrain. NN Archive replaces the old blob.
3. **Systemd autostart** — wrapper idea sketched, not built yet (cd → source ROS → source venv → source install → exec launch). Needs `loginctl enable-linger abcd`.
4. **TTS without internet** — `edge-playback` needs network. Fallback to `espeak-ng` or pre-rendered wav for offline boot.
6. **`log_window_frames` cleanup** — param declared in node + yaml but unused in current code path. Either remove, or wire it back (sliding-window majority log).

## Things cleaned this session

- **English voice commands → partial keyword detection**: English commands no longer require
  the full phrase. `speech_logger_node` now tokenizes English speech into whole words and
  matches single keywords (`money`/`banknote`/`cash`, `walk`/`forward`, `on`/`off`/`enable`/
  `disable`/`start`/`stop`/`resume`/`pause`, `hello`/`hi`/`hey`). Token matching (not substring)
  fixes a latent bug where "on" matches inside "navigati**on**". Thai still matched as substring.
- **GUI → web**: removed the OpenCV tactile preview window (`cv2.imshow`) entirely. Added `web_display_node` (stdlib HTTP server, port 8080) showing a live MJPEG video stream + current mode + latest alert text. `main_pipeline` publishes JPEG preview frames on `/oak/preview/compressed` (walk reuses the RGB tactile frame, money uses a small camera output); `web_display_node` re-serves them as MJPEG at `/stream`. `decision_audio_node` publishes `/oak/announcement`, and obstacle alerts are spoken/shown as friendly bilingual phrases ("ระวัง สิ่งกีดขวาง ด้านหน้า" / "Obstacle ahead").
- **Repo cleanup**: removed unused `oak_camera` + `oak_test` packages; deleted the 401 MB `tractile_model/` training tree (incl. a stray nested `.git`) — kept only the runtime `scripts/color_path_detect.py`; moved reference pipelines (`money_detect_ref_pipeline.py`, `test_yolo11_oak_new_reference.py`) into `oak_detectors/tools/`; deleted the now-unused `mobilenet-ssd...blob`; removed duplicate `src/{build,install,log}` artifacts. The runtime module is now just `main_pipeline.py` + `tractile_model/`.
- **Walk mode → depth-only**: dropped MobileNet-SSD entirely; obstacle detection is now stereo-depth left/center/right zones (see Walk pipeline details). Locked CAM_A manual focus so the lens stops hunting.
- Money pipeline migrated from `.blob` → NN Archive (`.tar.xz`) via `dai.NNArchive` + `DetectionNetwork.build`.
- Removed alphabetical-order labels + the `5_not_found` class.
- Fixed duplicate `declare_parameter("confidence_thresholds", ...)` (one was a typo for singular).
- `[det]` log throttled to 1 Hz (was per-frame).
- Money confirmation logic switched from 3-sec time window to 10-consecutive-frames count.
- Walk pipeline implemented end-to-end (was a stub).
- `main_pipeline.py` now branches `_poll_inference` on `self._current_mode` and tracks `self._active_mode_pipeline` to avoid running the wrong polling loop after a mode switch.
- `_cleanup_pipeline()` clears both money and walk queues + intrinsics.
- Render fix: money preview now updates even on frames without new detections (was early-returning).

## Memory

Project memory: `/home/abcd/.claude/projects/-home-abcd-blind-assist-ws/memory/`.
