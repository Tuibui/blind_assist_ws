#!/usr/bin/env python3
"""
Run OAK-D Lite blob inference and collect debug data for the next training round.

Headless: auto-save stores frames into auto_by_pred/<predicted_class>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


DEFAULT_LABEL_MAP = {
    0: "0_20",
    1: "1_50",
    2: "2_100",
    3: "3_500",
    4: "4_1000",
    5: "5_not_found",
}


@dataclass
class Prediction:
    index: int
    label: str
    score: float
    scores: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OAK-D Lite blob inference and save debug data."
    )
    parser.add_argument(
        "--blob-path",
        default="trained_models/best_custom_classifier_oak.blob",
        help="Path to compiled Myriad blob.",
    )
    parser.add_argument(
        "--class-map",
        default="trained_models/class_indices.json",
        help="Path to class_indices.json produced during training.",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug_capture",
        help="Directory for captured debug data.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="RGB preview width. Default: %(default)s",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=640,
        help="RGB preview height. Default: %(default)s",
    )
    parser.add_argument(
        "--nn-size",
        type=int,
        default=224,
        help="Neural network input size. Default: %(default)s",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Camera FPS. Default: %(default)s",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Auto-save frames below this confidence when auto mode is on.",
    )
    parser.add_argument(
        "--auto-save-interval",
        type=float,
        default=2.0,
        help="Minimum seconds between auto-saved debug frames.",
    )
    parser.add_argument(
        "--auto-save-mode",
        choices=("lowconf", "all"),
        default="lowconf",
        help="Auto-save all frames by interval, or only low-confidence ones. Default: %(default)s",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show. Default: %(default)s",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    if not path.is_file():
        return [DEFAULT_LABEL_MAP[i] for i in sorted(DEFAULT_LABEL_MAP)]

    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return [DEFAULT_LABEL_MAP[i] for i in sorted(DEFAULT_LABEL_MAP)]

    if all(isinstance(key, str) and key.isdigit() for key in data.keys()):
        pairs = sorted((int(key), value) for key, value in data.items())
        return [str(label) for _, label in pairs]

    reverse = {value: key for key, value in data.items()}
    pairs = sorted((index, label) for label, index in reverse.items())
    return [str(label) for _, label in pairs]


def ensure_device_access() -> None:
    devices = dai.Device.getAllAvailableDevices()
    if devices:
        return
    raise RuntimeError(
        "No available DepthAI devices. Reconnect the OAK-D Lite and confirm udev rules are installed."
    )


def build_runtime(
    pipeline: dai.Pipeline,
    blob_path: Path,
    preview_width: int,
    preview_height: int,
    nn_size: int,
    fps: float,
):
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    preview_output = cam.requestOutput(
        size=(preview_width, preview_height),
        type=dai.ImgFrame.Type.BGR888p,
        resizeMode=dai.ImgResizeMode.CROP,
        fps=fps,
    )
    nn_input = cam.requestOutput(
        size=(nn_size, nn_size),
        type=dai.ImgFrame.Type.RGB888p,
        resizeMode=dai.ImgResizeMode.CROP,
        fps=fps,
    )

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    nn_input.link(nn.input)
    return preview_output.createOutputQueue(), nn.out.createOutputQueue()


def decode_prediction(nn_packet: dai.NNData, class_names: list[str]) -> Prediction:
    scores = np.array(nn_packet.getFirstTensor(), dtype=np.float32).flatten().tolist()
    if not scores:
        raise RuntimeError("Neural network output is empty")

    pred_index = int(np.argmax(scores))
    label = class_names[pred_index] if pred_index < len(class_names) else str(pred_index)
    return Prediction(
        index=pred_index,
        label=label,
        score=float(scores[pred_index]),
        scores=[float(value) for value in scores],
    )


def rotate_180(frame: np.ndarray) -> np.ndarray:
    return cv2.rotate(frame, cv2.ROTATE_180)


def prepare_debug_dirs(debug_dir: Path, class_names: list[str]) -> dict[str, Path]:
    paths = {
        "review": debug_dir / "review",
        "auto_by_pred": debug_dir / "auto_by_pred",
        "misclassified": debug_dir / "misclassified",
        "relabel_root": debug_dir / "relabel",
        "negative_root": debug_dir / "negative",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        (paths["auto_by_pred"] / class_name).mkdir(parents=True, exist_ok=True)
        (paths["relabel_root"] / class_name).mkdir(parents=True, exist_ok=True)
        (paths["negative_root"] / f"{class_name}_negative").mkdir(parents=True, exist_ok=True)
    return paths


def save_frame(frame: np.ndarray, path: Path) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed to encode frame for {path}")
    path.write_bytes(encoded.tobytes())


def append_log(csv_path: Path, row: dict[str, str]) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "image_path",
                "pred_index",
                "pred_label",
                "pred_score",
                "scores_json",
                "capture_reason",
                "manual_label",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def capture_debug(
    frame: np.ndarray,
    prediction: Prediction,
    output_dir: Path,
    reason: str,
    csv_path: Path,
    manual_label: str = "",
) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{reason}_{prediction.label}_{prediction.score:.2f}.jpg"
    save_path = output_dir / filename
    save_frame(frame, save_path)
    append_log(
        csv_path,
        {
            "timestamp": timestamp,
            "image_path": str(save_path),
            "pred_index": str(prediction.index),
            "pred_label": prediction.label,
            "pred_score": f"{prediction.score:.6f}",
            "scores_json": json.dumps(prediction.scores),
            "capture_reason": reason,
            "manual_label": manual_label,
        },
    )
    return save_path


def main() -> int:
    args = parse_args()
    blob_path = Path(args.blob_path).expanduser().resolve()
    class_map_path = Path(args.class_map).expanduser().resolve()
    debug_dir = Path(args.debug_dir).expanduser().resolve()

    if not blob_path.is_file():
        raise FileNotFoundError(f"Blob not found: {blob_path}")

    class_names = load_class_names(class_map_path)
    debug_paths = prepare_debug_dirs(debug_dir, class_names)
    csv_path = debug_dir / "captures.csv"

    ensure_device_access()
    last_auto_save = 0.0

    try:
        with dai.Device() as device:
            with dai.Pipeline(device) as pipeline:
                preview_q, nn_q = build_runtime(
                    pipeline,
                    blob_path,
                    args.camera_width,
                    args.camera_height,
                    args.nn_size,
                    args.fps,
                )
                pipeline.start()

                while pipeline.isRunning():
                    frame_packet = preview_q.get()
                    nn_packet = nn_q.get()
                    frame = rotate_180(frame_packet.getCvFrame())
                    prediction = decode_prediction(nn_packet, class_names)

                    now = time.time()
                    should_auto_save = args.auto_save_mode == "all" or (
                        args.auto_save_mode == "lowconf"
                        and prediction.score < args.confidence_threshold
                    )
                    if (
                        should_auto_save
                        and now - last_auto_save >= args.auto_save_interval
                    ):
                        save_path = capture_debug(
                            frame,
                            prediction,
                            debug_paths["auto_by_pred"] / prediction.label,
                            args.auto_save_mode,
                            csv_path,
                        )
                        last_auto_save = now
                        print(f"[save] {prediction.label} ({prediction.score:.2f}) -> {save_path.name}")
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
