from pathlib import Path
import sys
import time

import depthai as dai


def main() -> int:
    blob_path = Path(__file__).resolve().parents[1] / "models" / "money" / "THB_bills.blob"
    if not blob_path.exists():
        print(f"Missing blob: {blob_path}", file=sys.stderr)
        return 1

    pipeline = dai.Pipeline()

    camera = pipeline.create(dai.node.ColorCamera)
    camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setInterleaved(False)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setPreviewSize(224, 224)
    camera.setFps(15)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    camera.preview.link(nn.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("nn")
    nn.out.link(xout.input)

    print(f"Using blob: {blob_path}")
    print("Waiting for OAK-D Lite inference output. Press Ctrl+C to stop.")

    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        last_report = time.monotonic()

        while True:
            packet = queue.get()
            values = list(packet.getFirstLayerFp16())
            now = time.monotonic()
            if now - last_report >= 0.3:
                print(f"raw_scores={values}")
                last_report = now


if __name__ == "__main__":
    raise SystemExit(main())
