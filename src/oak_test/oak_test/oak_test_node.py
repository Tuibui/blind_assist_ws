import importlib
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class OakTestNode(Node):
    def __init__(self) -> None:
        super().__init__("oak_test_node")

        self.declare_parameter("image_topic", "/oak/rgb/image_raw")
        self.declare_parameter("window_name", "OAK Camera Viewer")
        self.declare_parameter("show_fps", True)

        image_topic = self.get_parameter("image_topic").value
        self._window_name = self.get_parameter("window_name").value
        self._show_fps = bool(self.get_parameter("show_fps").value)

        self._cv2 = None
        self._window_initialized = False
        self._cv2_logged_error = False
        self._frame_count = 0
        self._fps_window_start = time.monotonic()
        self._fps_value = 0.0
        self._latest_frame = None
        self._display_timer = self.create_timer(1.0 / 30.0, self._update_viewer)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, image_topic, self._on_image, qos)
        self.get_logger().info(f"Subscribing to {image_topic} for display testing.")

    def _ensure_cv2(self) -> bool:
        if self._cv2 is not None:
            return True
        try:
            self._cv2 = importlib.import_module("cv2")
            return True
        except ImportError:
            if not self._cv2_logged_error:
                self.get_logger().error("opencv-python is not installed. pip install opencv-python.")
                self._cv2_logged_error = True
            return False

    def _decode_image(self, msg: Image):
        channels = 3 if msg.encoding in ("bgr8", "rgb8") else 1
        frame = np.frombuffer(msg.data, dtype=np.uint8)
        expected = msg.height * msg.step
        if frame.size < expected:
            self.get_logger().warning("Received truncated image buffer.")
            return None
        frame = frame[:expected].reshape((msg.height, msg.step))
        if channels == 3:
            frame = frame[:, : msg.width * 3].reshape((msg.height, msg.width, 3))
            if msg.encoding == "rgb8":
                frame = frame[:, :, ::-1]
        else:
            frame = frame[:, : msg.width]
        return frame

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps_value = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_window_start = now

    def _on_image(self, msg: Image) -> None:
        if not self._ensure_cv2():
            return

        frame = self._decode_image(msg)
        if frame is None:
            return

        self._update_fps()
        self._latest_frame = frame

    def _update_viewer(self) -> None:
        if not self._ensure_cv2():
            return
        if self._latest_frame is None:
            return

        frame = self._latest_frame
        display = frame.copy() if self._show_fps else frame
        if self._show_fps:
            self._cv2.putText(
                display,
                f"{self._fps_value:.1f} FPS",
                (12, 28),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                self._cv2.LINE_AA,
            )

        try:
            if not self._window_initialized:
                self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_AUTOSIZE)
                self._window_initialized = True
            self._cv2.imshow(self._window_name, display)
            key = self._cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                self.get_logger().info("Exit requested from viewer window.")
                rclpy.shutdown()
        except Exception as exc:
            self.get_logger().error(f"Viewer update failed: {exc}")
            self._window_initialized = False

    def destroy_node(self):
        if self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OakTestNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
