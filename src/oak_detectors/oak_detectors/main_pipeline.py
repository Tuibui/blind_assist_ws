from pathlib import Path
from collections import deque

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from oak_interfaces.msg import Detection2D, Detection2DArray, Mode


class MainPipelineNode(Node):
    def __init__(self) -> None:
        super().__init__("main_pipeline_node")

        share_dir = Path(get_package_share_directory("oak_detectors"))
        default_money_model = share_dir / "models" / "money" / "THB_bills.blob"

        self.declare_parameter("money_output_topic", "/money/detections")
        self.declare_parameter("navigation_output_topic", "/navigation/detections")
        self.declare_parameter("current_mode_topic", "/oak/current_mode")
        self.declare_parameter("money_model_path", str(default_money_model))
        self.declare_parameter("money_labels", ["100", "1000", "20", "50", "500"])
        self.declare_parameter("confidence_threshold", 0.70)
        self.declare_parameter("camera_fps", 15.0)
        self.declare_parameter("preview_width", 224)
        self.declare_parameter("preview_height", 224)
        self.declare_parameter("display_enabled", True)
        self.declare_parameter("display_window_name", "THB Bills Preview")
        self.declare_parameter("reconnect_period_sec", 2.0)
        self.declare_parameter("smoothing_window", 7)
        self.declare_parameter("stable_ratio_threshold", 0.60)
        self.declare_parameter("log_probabilities", True)
        self.declare_parameter("publish_empty_navigation", True)
        self.declare_parameter("log_model_usage", True)

        self._money_output_topic = str(self.get_parameter("money_output_topic").value)
        self._navigation_output_topic = str(self.get_parameter("navigation_output_topic").value)
        self._current_mode_topic = str(self.get_parameter("current_mode_topic").value)
        self._money_model_path = Path(str(self.get_parameter("money_model_path").value))
        self._money_labels = [str(label) for label in self.get_parameter("money_labels").value]
        self._confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self._camera_fps = float(self.get_parameter("camera_fps").value)
        self._preview_width = int(self.get_parameter("preview_width").value)
        self._preview_height = int(self.get_parameter("preview_height").value)
        self._display_enabled = bool(self.get_parameter("display_enabled").value)
        self._display_window_name = str(self.get_parameter("display_window_name").value)
        self._reconnect_period_sec = max(0.5, float(self.get_parameter("reconnect_period_sec").value))
        self._smoothing_window = max(3, int(self.get_parameter("smoothing_window").value))
        self._stable_ratio_threshold = min(1.0, max(0.5, float(self.get_parameter("stable_ratio_threshold").value)))
        self._log_probabilities = bool(self.get_parameter("log_probabilities").value)
        self._publish_empty_navigation = bool(self.get_parameter("publish_empty_navigation").value)
        self._log_model_usage = bool(self.get_parameter("log_model_usage").value)
        self._current_mode = Mode.MONEY
        self._dai = None
        self._cv2 = None
        self._pipeline = None
        self._device = None
        self._queue = None
        self._frame_queue = None
        self._display_available = False
        self._last_prediction_text = "No detection"
        self._pipeline_ready = False
        self._reconnect_due_ns = 0
        self._recent_probabilities = deque(maxlen=self._smoothing_window)
        self._last_probability_log_ns = 0

        self._money_publisher = self.create_publisher(Detection2DArray, self._money_output_topic, 10)
        self._navigation_publisher = self.create_publisher(Detection2DArray, self._navigation_output_topic, 10)
        self._mode_publisher = self.create_publisher(Mode, self._current_mode_topic, 10)

        self._log_model_state()
        self._setup_display()
        self._start_depthai_pipeline()
        self.create_timer(0.05, self._poll_inference)
        self.create_timer(1.0, self._publish_mode_heartbeat)

    def _log_model_state(self) -> None:
        exists = "ready" if self._money_model_path.exists() else "missing"
        self.get_logger().info(f"Money model: {self._money_model_path} ({exists})")

    def _import_depthai(self):
        if self._dai is not None:
            return self._dai
        try:
            import depthai as dai
        except ImportError as exc:
            raise RuntimeError(
                "depthai is not installed in the active environment. Activate venv before running main_pipeline."
            ) from exc
        self._dai = dai
        return dai

    def _setup_display(self) -> None:
        if not self._display_enabled:
            return
        try:
            import cv2
        except ImportError:
            self.get_logger().warning(
                "OpenCV is required for preview. Install opencv-python in venv."
            )
            raise RuntimeError("Missing opencv-python in venv.")
        self._cv2 = cv2
        self._display_available = True

    def _start_depthai_pipeline(self) -> None:
        if not self._money_model_path.exists():
            raise RuntimeError(f"Money model not found: {self._money_model_path}")

        self._cleanup_pipeline()
        dai = self._import_depthai()
        self._device = dai.Device()
        self._pipeline = dai.Pipeline(self._device)
        camera = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        nn = self._pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(self._money_model_path)
        nn_input = camera.requestOutput(
            (self._preview_width, self._preview_height),
            dai.ImgFrame.Type.RGB888p,
            dai.ImgResizeMode.CROP,
            float(self._camera_fps),
        )
        nn_input.link(nn.input)
        preview_output = None
        if self._display_enabled:
            preview_output = camera.requestOutput(
                (self._preview_width, self._preview_height),
                dai.ImgFrame.Type.BGR888i,
                dai.ImgResizeMode.CROP,
                float(self._camera_fps),
            )

        try:
            self._queue = nn.out.createOutputQueue(maxSize=4, blocking=False)
            if preview_output is not None:
                self._frame_queue = preview_output.createOutputQueue(maxSize=4, blocking=False)
            self._pipeline.start()
        except Exception as exc:
            self._cleanup_pipeline()
            raise RuntimeError(f"Failed to open OAK-D Lite for money pipeline: {exc}") from exc

        self._pipeline_ready = True
        self._reconnect_due_ns = 0
        if self._log_model_usage:
            self.get_logger().info("Money mode pipeline active: OAK-D Lite runs THB_bills.blob directly.")
            try:
                tensor_names = [tensor.name for tensor in nn.getOutputRefs()]
                self.get_logger().info(f"Money model outputs: {tensor_names}")
            except Exception:
                pass

    def _publish_mode_heartbeat(self) -> None:
        msg = Mode()
        msg.mode = Mode.MONEY
        self._mode_publisher.publish(msg)

    def _poll_inference(self) -> None:
        if not self._pipeline_ready:
            self._maybe_restart_pipeline()
            return
        if self._queue is None:
            return
        self._poll_preview_frame()

        try:
            packet = self._queue.tryGet()
        except Exception as exc:
            self._handle_pipeline_failure(f"Money inference queue closed: {exc}")
            return
        if packet is None:
            return

        try:
            tensor = packet.getFirstTensor()
            if hasattr(tensor, "flatten"):
                scores = tensor.flatten().tolist()
            else:
                scores = list(tensor)
        except Exception as exc:
            self.get_logger().error(f"Failed to decode money model output tensor: {exc}")
            return
        self._maybe_log_probabilities(scores)
        detections = self._scores_to_detections(scores)
        if detections:
            best = detections[0]
            self._last_prediction_text = f"{best.class_name} ({best.confidence:.2f})"
        else:
            self._last_prediction_text = "Unknown / move banknote closer"
        output = Detection2DArray()
        output.detections = detections
        self._money_publisher.publish(output)

        if self._publish_empty_navigation:
            self._navigation_publisher.publish(Detection2DArray())

    def _maybe_log_probabilities(self, scores) -> None:
        if not self._log_probabilities:
            return
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_probability_log_ns < int(1e9):
            return
        self._last_probability_log_ns = now_ns
        probabilities = [float(score) for score in scores[: len(self._money_labels)]]
        pairs = ", ".join(
            f"{label}:{prob:.3f}" for label, prob in zip(self._money_labels, probabilities)
        )
        self.get_logger().info(
            f"Money raw scores [RGB888p direct]: {pairs}"
        )

    def _poll_preview_frame(self) -> None:
        if self._frame_queue is None:
            return
        try:
            frame_packet = self._frame_queue.tryGet()
        except Exception as exc:
            self._handle_pipeline_failure(f"Preview queue closed: {exc}")
            return
        if frame_packet is None:
            return
        try:
            frame = frame_packet.getCvFrame()
        except Exception as exc:
            self._handle_pipeline_failure(f"Failed to decode preview frame: {exc}")
            return
        if frame is None:
            return
        self._render_preview(frame)

    def _render_preview(self, frame) -> None:
        if not self._display_available:
            return
        try:
            display_frame = frame.copy()
            self._cv2.putText(
                display_frame,
                self._last_prediction_text,
                (10, 24),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            self._cv2.imshow(self._display_window_name, display_frame)
            self._cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f"Preview display failed: {exc}")
            self._display_available = False

    def _handle_pipeline_failure(self, message: str) -> None:
        self.get_logger().error(message)
        self._cleanup_pipeline()
        self._reconnect_due_ns = self.get_clock().now().nanoseconds + int(self._reconnect_period_sec * 1e9)
        self._last_prediction_text = "Pipeline reconnecting"

    def _maybe_restart_pipeline(self) -> None:
        if self._reconnect_due_ns == 0:
            return
        if self.get_clock().now().nanoseconds < self._reconnect_due_ns:
            return
        self.get_logger().warning("Restarting money pipeline after device/queue failure.")
        try:
            self._start_depthai_pipeline()
        except Exception as exc:
            self.get_logger().error(f"Money pipeline restart failed: {exc}")
            self._reconnect_due_ns = self.get_clock().now().nanoseconds + int(self._reconnect_period_sec * 1e9)

    def _cleanup_pipeline(self) -> None:
        self._pipeline_ready = False
        self._queue = None
        self._frame_queue = None
        self._pipeline = None
        self._recent_probabilities.clear()
        if getattr(self, "_device", None) is not None:
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None

    def _scores_to_detections(self, scores):
        if not scores:
            return []

        probabilities = [float(score) for score in scores[: len(self._money_labels)]]
        if len(probabilities) != len(self._money_labels):
            return []
        self._recent_probabilities.append(probabilities)

        averaged = []
        for class_index in range(len(self._money_labels)):
            averaged.append(
                sum(frame_scores[class_index] for frame_scores in self._recent_probabilities)
                / len(self._recent_probabilities)
            )

        best_index = max(range(len(averaged)), key=lambda index: averaged[index])
        best_score = float(averaged[best_index])
        stable_votes = sum(
            1
            for frame_scores in self._recent_probabilities
            if max(range(len(frame_scores)), key=lambda index: frame_scores[index]) == best_index
        )
        stable_ratio = stable_votes / len(self._recent_probabilities)
        if best_score < self._confidence_threshold or stable_ratio < self._stable_ratio_threshold:
            return []

        detection = Detection2D()
        detection.class_name = self._money_labels[best_index]
        detection.confidence = best_score
        detection.x_min = 0.0
        detection.y_min = 0.0
        detection.x_max = 1.0
        detection.y_max = 1.0
        detection.center_x = 0.5
        detection.center_y = 0.5
        return [detection]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MainPipelineNode()
    try:
        rclpy.spin(node)
    finally:
        if getattr(node, "_cv2", None) is not None:
            try:
                node._cv2.destroyAllWindows()
            except Exception:
                pass
        node._cleanup_pipeline()
        node.destroy_node()
        rclpy.shutdown()
