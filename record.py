#!/usr/bin/env python3
"""Record a face dataset with the OAK-D-Lite (depthai 3.x).

Auto-captures one RGB frame every N seconds (default 3) and logs each capture
to the terminal. Images are saved under <out>/<name>/ with zero-padded,
continuing numbering so you can append to an existing dataset.

Usage (on the Raspberry Pi):
    python3 record.py --name alice
    python3 record.py --name bob --interval 3 --count 30
    python3 record.py --name carol --show          # add a preview window

Stop early with Ctrl-C; the capture count is printed on exit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai

REPO = Path(__file__).resolve().parent

# OAK-D-Lite is mounted upside down on this rig (matches the detector pipelines).
ROTATE = dai.CameraImageOrientation.ROTATE_180_DEG


def next_index(folder: Path, name: str) -> int:
    """Continue numbering from the highest existing <name>_####.jpg."""
    highest = 0
    for p in folder.glob(f"{name}_*.jpg"):
        stem = p.stem.rsplit("_", 1)[-1]
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="Person/label name; images go to <out>/<name>/")
    ap.add_argument("--out", default=str(REPO / "face_dataset"),
                    help="Dataset root (default: ./face_dataset)")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="Seconds between auto-captures (default: 3)")
    ap.add_argument("--count", type=int, default=0,
                    help="Stop after this many captures (0 = run until Ctrl-C)")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    ap.add_argument("--no-rotate", action="store_true",
                    help="Do not rotate 180 deg (camera mounted right-side up)")
    ap.add_argument("--show", action="store_true",
                    help="Show a live preview window (needs a display)")
    args = ap.parse_args()

    folder = Path(args.out) / args.name
    folder.mkdir(parents=True, exist_ok=True)
    idx = next_index(folder, args.name)

    print(f"[i] dataset : {folder}")
    print(f"[i] interval: {args.interval}s   "
          f"target: {args.count if args.count else 'unlimited'}   "
          f"start #: {idx:04d}")
    print(f"[i] resolution: {args.width}x{args.height}   "
          f"rotate180: {not args.no_rotate}")
    print("[i] Ctrl-C to stop.\n")

    with dai.Pipeline() as pipe:
        cam = pipe.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        if not args.no_rotate:
            cam.setImageOrientation(ROTATE)

        out = cam.requestOutput((args.width, args.height), dai.ImgFrame.Type.BGR888i)
        q = out.createOutputQueue(maxSize=4, blocking=False)

        pipe.start()
        saved = 0
        last_cap = time.monotonic() - args.interval  # capture first frame promptly
        try:
            while pipe.isRunning():
                pkt = q.get()
                frame = pkt.getCvFrame()

                if args.show:
                    cv2.imshow("record (q to quit)", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                now = time.monotonic()
                if now - last_cap < args.interval:
                    continue
                last_cap = now

                path = folder / f"{args.name}_{idx:04d}.jpg"
                cv2.imwrite(str(path), frame)
                h, w = frame.shape[:2]
                saved += 1
                print(f"[cap] {saved:3d}  {path.name}  ({w}x{h})")
                idx += 1

                if args.count and saved >= args.count:
                    break
        except KeyboardInterrupt:
            print()
        finally:
            if args.show:
                cv2.destroyAllWindows()

    print(f"\n[done] saved {saved} frame(s) to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
