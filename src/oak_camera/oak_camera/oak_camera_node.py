import importlib
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from oak_interfaces.msg import Mode


def build_mode_msg(mode_value: int) -> Mode:
    msg = Mode()
    msg.mode = mode_value
    return msg


class MockCameraBackend:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def apply_mode(self, mode_value: int) -> None:
        return None

    def get_frame(self):
        return bytes(self._width * self._height * 3)

    def is_started(self) -> bool:
        return True

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


@dataclass
class DepthAIState:
    pipeline: object
    device: object
    queue: object
    control_queue: object
    width: int
    height: int


class DepthAICameraBackend:
    def __init__(
        self,
        node: Node,
        width: int,
        height: int,
        walk_focus: int,
        money_focus: int,
        camera_fps: float,
    ) -> None:
        self._node = node
        self._width = width
        self._height = height
        self._walk_focus = walk_focus
        self._money_focus = money_focus
        self._camera_fps = camera_fps
        self._module = None
        self._state: Optional[DepthAIState] = None
        self._import_error_logged = False

    def _import_depthai(self) -> bool:
        if self._module is not None:
            return True
        try:
            self._module = importlib.import_module("depthai")
            return True
        except ImportError:
            if not self._import_error_logged:
                self._node.get_logger().error(
                    "depthai is not installed. Activate workspace venv and pip install depthai."
                )
                self._import_error_logged = True
            return False

    def start(self) -> bool:
        if self._state is not None:
            return True
        if not self._import_depthai():
            return False

        try:
            dai = self._module
            pipeline = dai.Pipeline(dai.Device())
            device = pipeline.getDefaultDevice()

            camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            capability = dai.ImgFrameCapability()
            capability.size.fixed((self._width, self._height))
            capability.fps.fixed(float(self._camera_fps))
            capability.type = dai.ImgFrame.Type.BGR888i
            host_output = camera.requestOutput(capability, True)

            queue = host_output.createOutputQueue(maxSize=1, blocking=False)
            control_queue = camera.inputControl.createInputQueue()
            pipeline.start()
            self._state = DepthAIState(
                pipeline=pipeline,
                device=device,
                queue=queue,
                control_queue=control_queue,
                width=self._width,
                height=self._height,
            )
            self._node.get_logger().info("Connected to OAK-D Lite through depthai Python.")
            return True
        except Exception as exc:
            self._node.get_logger().error(f"Failed to start depthai camera backend: {exc}")
            self.stop()
            return False

    def stop(self) -> None:
        if self._state is None:
            return
        try:
            self._state.device.close()
        except Exception:
            pass
        self._state = None

    def is_started(self) -> bool:
        return self._state is not None

    def apply_mode(self, mode_value: int) -> None:
        if self._state is None:
            return
        try:
            control = self._module.CameraControl()
            if mode_value == Mode.WALK:
                control.setAutoFocusMode(self._module.CameraControl.AutoFocusMode.OFF)
                control.setManualFocus(self._walk_focus)
            else:
                try:
                    control.setAutoFocusMode(self._module.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
                    control.setAutoFocusTrigger()
                except Exception:
                    control.setAutoFocusMode(self._module.CameraControl.AutoFocusMode.OFF)
                    control.setManualFocus(self._money_focus)
            self._state.control_queue.send(control)
        except Exception as exc:
            self._node.get_logger().warning(f"Failed to update camera focus mode: {exc}")

    def get_frame(self):
        if self._state is None:
            return None
        try:
            packet = self._state.queue.tryGet()
            if packet is None:
                return None
            while True:
                newer_packet = self._state.queue.tryGet()
                if newer_packet is None:
                    break
                packet = newer_packet
            frame = packet.getCvFrame()
            self._height = int(frame.shape[0])
            self._width = int(frame.shape[1])
            return frame
        except Exception as exc:
            self._node.get_logger().error(f"DepthAI frame acquisition failed: {exc}")
            self.stop()
            return None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


class OakCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("oak_camera_node")

        self.declare_parameter("mode_cmd_topic", "/oak/mode_cmd")
        self.declare_parameter("current_mode_topic", "/oak/current_mode")
        self.declare_parameter("image_topic", "/oak/rgb/image_raw")
        self.declare_parameter("publish_period_ms", 10)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("default_mode", int(Mode.WALK))
        self.declare_parameter("use_mock_camera", False)
        self.declare_parameter("frame_id", "oak_rgb_camera_frame")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 400)
        self.declare_parameter("walk_manual_focus", 130)
        self.declare_parameter("money_close_focus", 150)
        self.declare_parameter("reconnect_period_sec", 5.0)
        self.declare_parameter("display_gui", False)
        self.declare_parameter("window_name", "OAK Camera")
        self.declare_parameter("flip_vertical", True)

        self._mode_cmd_topic = self.get_parameter("mode_cmd_topic").value
        self._current_mode_topic = self.get_parameter("current_mode_topic").value
        self._image_topic = self.get_parameter("image_topic").value
        self._frame_id = self.get_parameter("frame_id").value
        self._display_gui = bool(self.get_parameter("display_gui").value)
        self._window_name = self.get_parameter("window_name").value
        self._flip_vertical = bool(self.get_parameter("flip_vertical").value)
        publish_period_sec = float(self.get_parameter("publish_period_ms").value) / 1000.0
        self._reconnect_period_sec = float(self.get_parameter("reconnect_period_sec").value)

        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        camera_fps = float(self.get_parameter("camera_fps").value)
        walk_focus = int(self.get_parameter("walk_manual_focus").value)
        money_focus = int(self.get_parameter("money_close_focus").value)

        default_mode = int(self.get_parameter("default_mode").value)
        self._current_mode = default_mode if default_mode in (Mode.WALK, Mode.MONEY) else Mode.WALK
        self._last_reconnect_attempt = 0.0
        self._cv2 = None
        self._cv2_logged_error = False
        self._window_initialized = False
        self._running = True

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._mode_publisher = self.create_publisher(Mode, self._current_mode_topic, 10)
        self._image_publisher = self.create_publisher(Image, self._image_topic, image_qos)
        self.create_subscription(Mode, self._mode_cmd_topic, self._on_mode_cmd, 10)

        use_mock = bool(self.get_parameter("use_mock_camera").value)
        if use_mock:
            self._backend = MockCameraBackend(width=width, height=height)
            self._backend.start()
            self.get_logger().info("Using mock camera backend.")
        else:
            self._backend = DepthAICameraBackend(
                node=self,
                width=width,
                height=height,
                walk_focus=walk_focus,
                money_focus=money_focus,
                camera_fps=camera_fps,
            )

        self._backend.apply_mode(self._current_mode)
        self._mode_timer = self.create_timer(max(publish_period_sec, 0.1), self._publish_mode_heartbeat)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _ensure_cv2(self) -> bool:
        if not self._display_gui:
            return False
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

    def _show_frame(self, frame) -> None:
        if not self._ensure_cv2():
            return
        try:
            if not self._window_initialized:
                self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_AUTOSIZE)
                self._window_initialized = True
            self._cv2.imshow(self._window_name, frame)
            key = self._cv2.waitKey(1)
            if key in (27, ord("q"), ord("Q")):
                self.get_logger().info("Exit requested from camera display window.")
                rclpy.shutdown()
        except Exception as exc:
            self.get_logger().error(f"Camera GUI update failed: {exc}")
            self._window_initialized = False

    def _on_mode_cmd(self, msg: Mode) -> None:
        if msg.mode not in (Mode.WALK, Mode.MONEY):
            self.get_logger().warning(f"Ignoring unsupported mode value: {msg.mode}")
            return
        requested_mode_name = "WALK" if msg.mode == Mode.WALK else "MONEY"
        self.get_logger().info(f"Received mode command: {requested_mode_name} ({msg.mode}).")
        if msg.mode == self._current_mode:
            self.get_logger().info(f"Camera mode already {requested_mode_name}; no focus change needed.")
            return
        self._current_mode = msg.mode
        self._backend.apply_mode(self._current_mode)
        self._mode_publisher.publish(build_mode_msg(self._current_mode))
        mode_name = "WALK" if self._current_mode == Mode.WALK else "MONEY"
        self.get_logger().info(f"Camera mode updated to {mode_name}.")

    def _publish_mode_heartbeat(self) -> None:
        self._mode_publisher.publish(build_mode_msg(self._current_mode))

    def _ensure_backend_started(self) -> bool:
        if self._backend.is_started():
            return True
        now = time.monotonic()
        if now - self._last_reconnect_attempt < self._reconnect_period_sec:
            return False
        self._last_reconnect_attempt = now
        started = self._backend.start()
        if started:
            self._backend.apply_mode(self._current_mode)
        return started

    def _capture_loop(self) -> None:
        while self._running and rclpy.ok():
            if not self._ensure_backend_started():
                time.sleep(0.02)
                continue

            frame = self._backend.get_frame()
            if frame is None:
                time.sleep(0.002)
                continue

            if self._flip_vertical:
                frame = self._flip_frame(frame)

            if not isinstance(frame, bytes):
                self._show_frame(frame)
            self._publish_image(frame)

    def _flip_frame(self, frame):
        if isinstance(frame, bytes):
            return frame
        return frame[::-1, :, :]

    def _publish_image(self, frame) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = int(self._backend.height)
        msg.width = int(self._backend.width)
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        if isinstance(frame, bytes):
            msg.data = frame
        else:
            msg.data = frame.tobytes()
        self._image_publisher.publish(msg)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = OakCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        if hasattr(node, "_capture_thread"):
            node._capture_thread.join(timeout=1.0)
        if getattr(node, "_cv2", None) is not None:
            try:
                node._cv2.destroyAllWindows()
            except Exception:
                pass
        if hasattr(node, "_backend"):
            node._backend.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
