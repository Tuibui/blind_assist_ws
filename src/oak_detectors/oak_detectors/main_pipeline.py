"""node กล้องและตรวจจับหลักของระบบ (ทำงานได้ทีละโหมด)

OAK-D Lite เป็นกล้องตัวเดียว จึงรันได้ทีละ "ไปป์ไลน์" เท่านั้น node นี้จึงเป็น
เจ้าของไปป์ไลน์ทั้ง 3 โหมดและสลับไปมาตามคำสั่งโหมด (/oak/mode_cmd):

  • โหมดเดิน (WALK)  : ใช้ stereo depth อย่างเดียว แบ่งภาพ depth เป็น 3 ช่อง
                       ซ้าย/กลาง/ขวา ช่องไหนมีพิกเซลใกล้มากพอก็เตือนว่ามีสิ่งกีดขวาง
                       ทางนั้น — ไม่ใช้ neural network จึงจับได้ทุกอย่าง (กำแพง เสา
                       กระจก กล่อง) พร้อมกันนั้นยังตรวจ "ทางเดิน (tactile)" จากภาพ RGB
  • โหมดเงิน (MONEY) : ใช้ DetectionNetwork + NN Archive จำแนกธนบัตรไทย
  • โหมดใบหน้า (FACE): โครงสร้างเดียวกับโหมดเงิน แต่ใช้โมเดลจำใบหน้าเพื่อน

เมื่อสลับโหมด _cleanup_pipeline() จะปิดไปป์ไลน์เดิมและสร้างอันใหม่ (ใช้เวลา ~1-2 วิ)
"""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from oak_interfaces.msg import Detection2D, Detection2DArray, Mode

# ตัวตรวจ "ทางเดิน (tactile path)" จากสี — ใช้ร่วมในโหมดเดิน
from oak_detectors.tractile_model.scripts.color_path_detect import (
    process as tactile_process,
)


class MainPipelineNode(Node):
    """node หลัก: ถือไปป์ไลน์กล้องทั้ง 3 โหมด สลับโหมด และส่งผลตรวจจับออกเป็น topic"""

    def __init__(self) -> None:
        super().__init__("main_pipeline_node")

        # path เริ่มต้นของโมเดล (อยู่ในโฟลเดอร์ share/models ที่ติดตั้งไว้)
        share_dir = Path(get_package_share_directory("oak_detectors"))
        default_money_model = share_dir / "models" / "money" / "best.rvc2.tar.xz"
        default_face_model = share_dir / "models" / "myfriend_face" / "best.rvc2.tar.xz"

        # ---- ประกาศพารามิเตอร์ทั้งหมด (ปรับค่าได้ผ่านไฟล์ main_pipeline.yaml) ----
        self.declare_parameter("money_output_topic", "/money/detections")
        self.declare_parameter("navigation_output_topic", "/navigation/detections")
        self.declare_parameter("face_output_topic", "/face/detections")
        self.declare_parameter("current_mode_topic", "/oak/current_mode")
        self.declare_parameter("money_model_path", str(default_money_model))
        self.declare_parameter(
            "money_labels",
            ["20", "50", "100", "500", "1000"],
        )
        # โหมดจำใบหน้า (FACE): ใช้ NN Archive รูปแบบเดียวกับโหมดเงิน คลาสคือชื่อเพื่อน
        # ที่เทรนโมเดลมา — ตรวจเจอใครก็พูดชื่อคนนั้น
        self.declare_parameter("face_model_path", str(default_face_model))
        self.declare_parameter("face_labels", ["Tee", "Ping"])
        self.declare_parameter("face_confidence_threshold", 0.5)
        self.declare_parameter("confidence_threshold", 0.2)
        self.declare_parameter("confidence_thresholds", [0.6, 0.6, 0.20, 0.4, 0.45])
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("nn_input_size", 416)
        self.declare_parameter("rotate_preview_180", True)
        self.declare_parameter("reconnect_period_sec", 2.0)
        self.declare_parameter("publish_empty_navigation", True)
        self.declare_parameter("log_model_usage", True)
        self.declare_parameter("mode_cmd_topic", "/oak/mode_cmd")
        self.declare_parameter("default_mode", "walk")
        self.declare_parameter("log_window_frames", 2)
        # สิ่งกีดขวาง = มีอะไรอยู่ใกล้กว่าระยะนี้ในช่องนั้น (ระยะเป็นมิลลิเมตร)
        self.declare_parameter("walk_obstacle_z_max_mm", 1500.0)
        # ตำแหน่งโฟกัสเลนส์แบบล็อกค่าในโหมดเดิน เพื่อไม่ให้ CAM_A วิ่งออโต้โฟกัส
        # (เลนส์จะส่ายไปมา) 0 = ไกล/อนันต์, 255 = ระยะใกล้สุด (มาโคร)
        self.declare_parameter("walk_manual_focus", 15)
        # ตรวจสิ่งกีดขวางด้วย depth ล้วน (ไม่มี NN): แบ่งภาพ depth เป็นช่องซ้าย/กลาง/ขวา
        # แล้วเตือนเมื่อมีพิกเซลใกล้มากพอ
        # ค่านี้คือ "สัดส่วนพิกเซลในช่อง" ที่ต้องใกล้กว่า z_max จึงจะนับว่าเป็นสิ่งกีดขวาง
        # (กรอง noise ของ depth / พิกเซลหลงทิ้งไม่กี่จุดออก)
        self.declare_parameter("walk_obstacle_min_pixel_frac", 0.04)
        # ขอบเขตแนวตั้งที่สนใจ (เป็นสัดส่วนของความสูงภาพ) ใช้วิเคราะห์สิ่งกีดขวาง
        # — ตัดเพดาน/พื้นออกเพื่อไม่ให้ไปกระตุ้นเตือนผิด ๆ
        self.declare_parameter("walk_roi_top_frac", 0.1)
        self.declare_parameter("walk_roi_bottom_frac", 0.9)
        # ภาพพรีวิวสดที่ส่งไปแสดงบนหน้าเว็บ เป็น JPEG (sensor_msgs/CompressedImage)
        # โหมดเดินสตรีมภาพจากกล้อง RGB ส่วนโหมดเงินสตรีมภาพเล็ก ๆ จากกล้อง
        self.declare_parameter("preview_enabled", True)
        self.declare_parameter("preview_topic", "/oak/preview/compressed")
        self.declare_parameter("preview_jpeg_quality", 50)
        self.declare_parameter("preview_max_width", 480)
        # ตรวจ "ทางเดิน (tactile)" จากสี — ทำงานควบคู่ไปกับการตรวจสิ่งกีดขวางในโหมดเดิน
        self.declare_parameter("tactile_enabled", True)
        # คำสั่งเสียง "เปิด/ปิดระบบนำทาง" — สวิตช์ tactile ขณะรันโดยไม่ออกจากโหมดเดิน
        # (สิ่งกีดขวางยังตรวจอยู่) std_msgs/Bool: true=เปิดนำทาง / false=ปิด
        self.declare_parameter("nav_cmd_topic", "/oak/nav_cmd")
        self.declare_parameter("tactile_output_topic", "/tactile/path")
        self.declare_parameter("tactile_input_width", 640)
        self.declare_parameter("tactile_input_height", 360)
        self.declare_parameter("tactile_h_lo", 5)
        self.declare_parameter("tactile_h_hi", 20)
        self.declare_parameter("tactile_s_lo", 25)
        self.declare_parameter("tactile_v_lo", 85)
        self.declare_parameter("tactile_horizon", 0.20)
        self.declare_parameter("tactile_bottom", 0.30)
        self.declare_parameter("tactile_center", 0.30)
        self.declare_parameter("tactile_dead_off", 0.10)
        self.declare_parameter("tactile_sharp_off", 0.30)
        # เมื่อเจอสิ่งกีดขวางจาก depth ให้ "พัก" การตรวจทางเดินไว้กี่วินาที
        # เหตุผล: คนยืนใกล้ ๆ จะบังภาพช่วงล่าง ขา/เสื้อผ้าอาจถูกเข้าใจผิดว่าเป็นทางเดิน
        # ระหว่างที่มีสิ่งกีดขวางเราจึงเชื่อการเตือนสิ่งกีดขวางแทน และหยุดส่งทางเดิน
        # (ที่น่าจะผิด) ออกไป ใส่ 0 เพื่อปิดการพักนี้
        self.declare_parameter("tactile_obstacle_pause_sec", 2.0)
        # ชั่วคราว/ดีบัก: ป้อนภาพตรวจทางเดินในโหมดเดินจาก "ไฟล์วิดีโอ" แทนกล้อง OAK
        # ถ้าใส่ path ไว้ โหมดเดินจะข้ามกล้อง OAK ทั้งหมด (ไม่มีการตรวจสิ่งกีดขวางจาก depth)
        # แล้วรันตัวตรวจทางเดินบนเฟรมจากวิดีโอ วนกลับไปต้นไฟล์เมื่อจบ
        # ปล่อยเป็น "" สำหรับการใช้งานกล้องจริงตามปกติ
        self.declare_parameter("tactile_video_source", "")

        self._money_output_topic = str(self.get_parameter("money_output_topic").value)
        self._navigation_output_topic = str(self.get_parameter("navigation_output_topic").value)
        self._face_output_topic = str(self.get_parameter("face_output_topic").value)
        self._current_mode_topic = str(self.get_parameter("current_mode_topic").value)
        self._money_model_path = Path(str(self.get_parameter("money_model_path").value))
        self._money_labels = [str(label) for label in self.get_parameter("money_labels").value]
        self._face_model_path = Path(str(self.get_parameter("face_model_path").value))
        self._face_labels = [str(label) for label in self.get_parameter("face_labels").value]
        self._face_conf = float(self.get_parameter("face_confidence_threshold").value)
        self._confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        raw_per_class = list(self.get_parameter("confidence_thresholds").value or [])
        self._per_class_thresholds = [
            float(t) if t is not None and float(t) >= 0.0 else self._confidence_threshold
            for t in raw_per_class
        ]
        if len(self._per_class_thresholds) != len(self._money_labels):
            self.get_logger().warning(
                f"confidence_thresholds length {len(self._per_class_thresholds)} != labels "
                f"{len(self._money_labels)}; falling back to global confidence_threshold for all classes."
            )
            self._per_class_thresholds = [self._confidence_threshold] * len(self._money_labels)
        self._device_conf_floor = min(self._per_class_thresholds) if self._per_class_thresholds else self._confidence_threshold
        self.get_logger().info(
            "Per-class thresholds: "
            + ", ".join(f"{lbl}={thr:.2f}" for lbl, thr in zip(self._money_labels, self._per_class_thresholds))
        )
        self._camera_fps = float(self.get_parameter("camera_fps").value)
        self._nn_input_size = int(self.get_parameter("nn_input_size").value)
        self._rotate_preview_180 = bool(self.get_parameter("rotate_preview_180").value)
        self._reconnect_period_sec = max(0.5, float(self.get_parameter("reconnect_period_sec").value))
        self._publish_empty_navigation = bool(self.get_parameter("publish_empty_navigation").value)
        self._log_model_usage = bool(self.get_parameter("log_model_usage").value)
        self._mode_cmd_topic = str(self.get_parameter("mode_cmd_topic").value)
        default_mode_str = str(self.get_parameter("default_mode").value).strip().lower()
        if default_mode_str == "money":
            self._current_mode = Mode.MONEY
        elif default_mode_str == "face":
            self._current_mode = Mode.FACE
        else:
            self._current_mode = Mode.WALK

        self._dai = None
        self._pipeline = None
        self._device = None
        self._det_queue = None
        self._archive_names = None
        self._last_prediction_text = "No detection"
        self._pipeline_ready = False
        self._log_window_frames = max(1, int(self.get_parameter("log_window_frames").value))
        self._label_window: list[str] = []
        self._reconnect_due_ns = 0

        self._walk_obstacle_z_max_mm = float(self.get_parameter("walk_obstacle_z_max_mm").value)
        self._walk_manual_focus = int(self.get_parameter("walk_manual_focus").value)
        self._walk_obstacle_min_frac = float(self.get_parameter("walk_obstacle_min_pixel_frac").value)
        self._walk_roi_top_frac = float(self.get_parameter("walk_roi_top_frac").value)
        self._walk_roi_bottom_frac = float(self.get_parameter("walk_roi_bottom_frac").value)
        self._walk_depth_queue = None
        self._active_mode_pipeline = None  # ไปป์ไลน์ที่กำลังทำงาน: "walk" | "money" | "face" | None

        self._preview_enabled = bool(self.get_parameter("preview_enabled").value)
        self._preview_topic = str(self.get_parameter("preview_topic").value)
        self._preview_jpeg_quality = int(self.get_parameter("preview_jpeg_quality").value)
        self._preview_max_width = int(self.get_parameter("preview_max_width").value)
        self._det_preview_queue = None
        # ตั้งค่าตอนเริ่มไปป์ไลน์ตรวจจับ (money/face) เพื่อให้โค้ด polling ใช้ร่วมกันได้:
        # บอกว่าผลตรวจจับจะส่งออกทาง publisher ไหน และใช้ threshold ต่อคลาสชุดใด
        self._active_det_publisher = None
        self._active_thresholds: list[float] = []

        self._tactile_enabled = bool(self.get_parameter("tactile_enabled").value)
        self._nav_cmd_topic = str(self.get_parameter("nav_cmd_topic").value)
        # สวิตช์ระบบนำทาง (tactile) ขณะรัน: เริ่มต้นเปิดเสมอ (= ค่า default ของโหมดเดิน)
        # คำสั่งเสียง "ปิดระบบนำทาง" ตั้งเป็น False / "เปิดระบบนำทาง" ตั้งกลับ True
        # และจะถูกรีเซ็ตเป็น True ทุกครั้งที่เข้าโหมดเดินใหม่
        self._tactile_runtime_on = True
        self._tactile_output_topic = str(self.get_parameter("tactile_output_topic").value)
        self._tactile_in_w = int(self.get_parameter("tactile_input_width").value)
        self._tactile_in_h = int(self.get_parameter("tactile_input_height").value)
        self._tactile_args = SimpleNamespace(
            h_lo=int(self.get_parameter("tactile_h_lo").value),
            h_hi=int(self.get_parameter("tactile_h_hi").value),
            s_lo=int(self.get_parameter("tactile_s_lo").value),
            v_lo=int(self.get_parameter("tactile_v_lo").value),
            horizon=float(self.get_parameter("tactile_horizon").value),
            bottom=float(self.get_parameter("tactile_bottom").value),
            center=float(self.get_parameter("tactile_center").value),
            dead_off=float(self.get_parameter("tactile_dead_off").value),
            sharp_off=float(self.get_parameter("tactile_sharp_off").value),
            show_rejected=False,
        )
        self._tactile_queue = None
        self._tactile_obstacle_pause_sec = float(
            self.get_parameter("tactile_obstacle_pause_sec").value
        )
        self._tactile_pause_until_ns = 0   # ข้ามการตรวจทางเดินจนถึงเวลานี้ (ระหว่างมีสิ่งกีดขวางใกล้)
        self._tactile_video_source = str(self.get_parameter("tactile_video_source").value).strip()
        self._walk_video_cap = None
        self._walk_video_mode = False

        # ---- สร้าง publisher (ส่งออก) / subscription (รับเข้า) ----
        self._money_publisher = self.create_publisher(Detection2DArray, self._money_output_topic, 10)
        self._face_publisher = self.create_publisher(Detection2DArray, self._face_output_topic, 10)
        self._navigation_publisher = self.create_publisher(Detection2DArray, self._navigation_output_topic, 10)
        self._tactile_publisher = self.create_publisher(Detection2DArray, self._tactile_output_topic, 10)
        self._mode_publisher = self.create_publisher(Mode, self._current_mode_topic, 10)
        self._preview_publisher = (
            self.create_publisher(CompressedImage, self._preview_topic, 1)
            if self._preview_enabled else None
        )
        self.create_subscription(Mode, self._mode_cmd_topic, self._on_mode_cmd, 10)
        self.create_subscription(Bool, self._nav_cmd_topic, self._on_nav_cmd, 10)

        # เริ่มไปป์ไลน์ตามโหมดเริ่มต้นทันทีตอนเปิดเครื่อง
        self._log_model_state()
        try:
            if self._current_mode == Mode.MONEY:
                self._start_money_pipeline()
            elif self._current_mode == Mode.FACE:
                self._start_face_pipeline()
            else:
                self._start_walk_pipeline()
        except Exception as exc:
            self.get_logger().error(f"Failed to start pipeline at boot: {exc}")
        # timer หลัก: ดึงผล inference ทุก 50 มิลลิวินาที (~20 Hz)
        self.create_timer(0.05, self._poll_inference)
        # timer ย่อย: ประกาศโหมดปัจจุบันซ้ำทุก 1 วิ (heartbeat ให้ node อื่นรู้สถานะ)
        self.create_timer(1.0, self._publish_mode_heartbeat)

    def _log_model_state(self) -> None:
        """log บอกว่าพบไฟล์โมเดล (เงิน/ใบหน้า) หรือไม่ ตอนเริ่มต้น"""
        money_state = "ready" if self._money_model_path.exists() else "missing"
        face_state = "ready" if self._face_model_path.exists() else "missing"
        self.get_logger().info(f"Money NN Archive: {self._money_model_path} ({money_state})")
        self.get_logger().info(f"Face NN Archive: {self._face_model_path} ({face_state})")
        self.get_logger().info("Walk mode: depth-only obstacle zones (no model needed)")

    def _import_depthai(self):
        """import ไลบรารี depthai แบบ lazy (โหลดครั้งแรกครั้งเดียว แล้ว cache ไว้)"""
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

    def _start_money_pipeline(self) -> None:
        """เริ่มไปป์ไลน์โหมดเงิน: ใช้โมเดลธนบัตร + threshold ต่อคลาส"""
        self._start_detection_pipeline(
            kind="money",
            model_path=self._money_model_path,
            labels=self._money_labels,
            thresholds=self._per_class_thresholds,
            publisher=self._money_publisher,
        )

    def _start_face_pipeline(self) -> None:
        """เริ่มไปป์ไลน์โหมดใบหน้า: ใช้โมเดลจำเพื่อน + threshold ค่าเดียวทุกคลาส"""
        self._start_detection_pipeline(
            kind="face",
            model_path=self._face_model_path,
            labels=self._face_labels,
            thresholds=[self._face_conf] * len(self._face_labels),
            publisher=self._face_publisher,
        )

    def _start_detection_pipeline(self, kind, model_path, labels, thresholds, publisher) -> None:
        """สร้างไปป์ไลน์ DetectionNetwork จาก NN Archive (ใช้กล้อง CAM_A อย่างเดียว)

        โหมดเงินและโหมดใบหน้ามีโครงสร้างเหมือนกันเป๊ะ — กล้องเดียวกัน รูปแบบ
        YOLO archive เดียวกัน — จึงใช้ฟังก์ชันสร้างตัวนี้ร่วมกัน ต่างกันแค่ตัวโมเดล,
        labels/thresholds และ topic ที่จะส่งผลตรวจจับออกไป
        """
        if not model_path.exists():
            raise RuntimeError(f"{kind} NN Archive not found: {model_path}")

        self._cleanup_pipeline()
        dai = self._import_depthai()
        self._device = dai.Device()
        self._pipeline = dai.Pipeline(self._device)
        camera = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        if self._rotate_preview_180:
            camera.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        archive = dai.NNArchive(str(model_path))
        det = self._pipeline.create(dai.node.DetectionNetwork).build(
            camera, archive,
            fps=float(self._camera_fps),
            resizeMode=dai.ImgResizeMode.CROP,
        )
        conf_floor = min(thresholds) if thresholds else self._confidence_threshold
        det.setConfidenceThreshold(float(conf_floor))

        preview_out = None
        if self._preview_enabled:
            try:
                preview_out = camera.requestOutput(
                    (640, 360), dai.ImgFrame.Type.BGR888i, fps=float(self._camera_fps)
                )
            except Exception as exc:
                self.get_logger().warning(f"{kind} preview output unavailable: {exc}")
                preview_out = None

        names_from_archive = det.getClasses() or []
        if names_from_archive and len(names_from_archive) == len(labels):
            self._archive_names = {i: n for i, n in enumerate(names_from_archive)}
        else:
            self._archive_names = {i: n for i, n in enumerate(labels)}

        try:
            self._det_queue = det.out.createOutputQueue(maxSize=4, blocking=False)
            if preview_out is not None:
                self._det_preview_queue = preview_out.createOutputQueue(maxSize=1, blocking=False)
            else:
                self._det_preview_queue = None
            self._pipeline.start()
        except Exception as exc:
            self._cleanup_pipeline()
            raise RuntimeError(f"Failed to open OAK-D Lite for {kind} pipeline: {exc}") from exc

        self._active_det_publisher = publisher
        self._active_thresholds = list(thresholds)
        self._pipeline_ready = True
        self._active_mode_pipeline = kind
        self._reconnect_due_ns = 0
        if self._log_model_usage:
            self.get_logger().info(
                f"{kind} mode pipeline active: DetectionNetwork on NN Archive, classes={self._archive_names}, "
                f"conf_floor={conf_floor:.2f}"
            )

    def _start_walk_pipeline(self) -> None:
        """เริ่มไปป์ไลน์โหมดเดิน: stereo depth (ตรวจสิ่งกีดขวาง) + กล้อง RGB (ตรวจทางเดิน)
        ถ้ามีการตั้ง tactile_video_source ไว้ จะเปลี่ยนไปใช้โหมดวิดีโอแทน"""
        if self._tactile_video_source:
            self._start_walk_video_pipeline()
            return
        self._cleanup_pipeline()
        dai = self._import_depthai()
        self._device = dai.Device()
        self._pipeline = dai.Pipeline(self._device)

        cam_rgb = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        if self._rotate_preview_180:
            cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        # ล็อกโฟกัสเลนส์ CAM_A ไว้คงที่ ไม่งั้นมันจะวิ่งออโต้โฟกัสและเลนส์ส่ายเห็นได้ชัด
        # ระหว่างโหมดเดิน — CAM_A เก็บไว้แค่เพื่อตรวจทางเดิน (tactile) และเป็นภาพอ้างอิง
        # สำหรับจัดตำแหน่ง depth ส่วนการตรวจสิ่งกีดขวางใช้ depth ล้วน (ไม่มี neural network)
        walk_control_queue = cam_rgb.inputControl.createInputQueue()

        tactile_out = None
        if self._tactile_enabled:
            try:
                tactile_out = cam_rgb.requestOutput(
                    (self._tactile_in_w, self._tactile_in_h),
                    dai.ImgFrame.Type.BGR888i,
                    fps=float(self._camera_fps),
                )
            except Exception as exc:
                self.get_logger().warning(f"Tactile camera output unavailable: {exc}")
                tactile_out = None

        left = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self._pipeline.create(dai.node.StereoDepth)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        try:
            stereo.setOutputSize(640, 400)
        except Exception:
            pass
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        except Exception:
            pass
        left.requestOutput((640, 400), dai.ImgFrame.Type.NV12, fps=float(self._camera_fps)).link(stereo.left)
        right.requestOutput((640, 400), dai.ImgFrame.Type.NV12, fps=float(self._camera_fps)).link(stereo.right)

        try:
            self._walk_depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
            if tactile_out is not None:
                self._tactile_queue = tactile_out.createOutputQueue(maxSize=2, blocking=False)
            else:
                self._tactile_queue = None
            self._pipeline.start()
        except Exception as exc:
            self._cleanup_pipeline()
            raise RuntimeError(f"Failed to open OAK-D Lite for walk pipeline: {exc}") from exc

        try:
            control = dai.CameraControl()
            control.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
            control.setManualFocus(self._walk_manual_focus)
            walk_control_queue.send(control)
        except Exception as exc:
            self.get_logger().warning(f"Could not set walk manual focus: {exc}")

        self._pipeline_ready = True
        self._active_mode_pipeline = "walk"
        self._reconnect_due_ns = 0
        self.get_logger().info(
            f"Walk pipeline active: depth-only obstacle zones (left/center/right), "
            f"z<={self._walk_obstacle_z_max_mm:.0f}mm, min_frac={self._walk_obstacle_min_frac:.2f}, "
            f"roi_rows=[{self._walk_roi_top_frac:.2f},{self._walk_roi_bottom_frac:.2f}]"
        )

    def _start_walk_video_pipeline(self) -> None:
        """ชั่วคราว: โหมดเดินที่ตรวจทางเดินจากไฟล์วิดีโอ (ไม่ใช้ OAK)

        อ่านเฟรมจาก self._tactile_video_source แล้วรันตัวตรวจทางเดินจากสีตัวเดียวกับ
        ที่ไปป์ไลน์เดินแบบสดใช้ ส่งผลออก /tactile/path และภาพพรีวิวบนเว็บ
        การตรวจสิ่งกีดขวางจาก depth จะถูกข้าม (วิดีโอ 2 มิติไม่มีข้อมูล depth)
        และวนกลับไปต้นไฟล์เมื่อเล่นจบ
        """
        self._cleanup_pipeline()
        import cv2

        cap = cv2.VideoCapture(self._tactile_video_source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open tactile_video_source: {self._tactile_video_source}"
            )
        self._walk_video_cap = cap
        self._walk_video_mode = True
        self._pipeline_ready = True
        self._active_mode_pipeline = "walk"
        self._reconnect_due_ns = 0
        self.get_logger().warning(
            f"Walk pipeline in VIDEO mode (no OAK, no depth obstacles): "
            f"{self._tactile_video_source} — tactile detection only."
        )

    def _publish_mode_heartbeat(self) -> None:
        """ประกาศโหมดปัจจุบันออก topic เป็นระยะ ให้ node อื่นรู้สถานะเสมอ"""
        msg = Mode()
        msg.mode = self._current_mode
        self._mode_publisher.publish(msg)

    def _on_nav_cmd(self, msg: Bool) -> None:
        """คำสั่งเสียง "เปิด/ปิดระบบนำทาง" — สลับ tactile ขณะรันโดยไม่ออกจากโหมดเดิน
        ปิดอยู่ = ไม่ตรวจ/ไม่ส่งทางเดิน (แต่ยังตรวจสิ่งกีดขวางตามปกติ)"""
        self._tactile_runtime_on = bool(msg.data)
        state = "ON" if self._tactile_runtime_on else "OFF"
        self.get_logger().info(f"Navigation (tactile) runtime switch: {state}")

    def _on_mode_cmd(self, msg: Mode) -> None:
        """รับคำสั่งเปลี่ยนโหมด แล้วปิดไปป์ไลน์เดิม/เปิดไปป์ไลน์ใหม่ตามโหมดที่สั่ง"""
        if msg.mode not in (Mode.WALK, Mode.MONEY, Mode.FACE):
            return
        # เข้าโหมดเดิน = รีเซ็ตระบบนำทางกลับเป็น "เปิด" เสมอ (ค่า default) แม้สั่ง
        # "เดินต่อไป" ทั้งที่อยู่ในโหมดเดินอยู่แล้ว ก็ถือเป็นการเรียกนำทางกลับมา
        if msg.mode == Mode.WALK:
            self._tactile_runtime_on = True
        if msg.mode == self._current_mode:
            return
        self._current_mode = msg.mode
        if msg.mode == Mode.MONEY:
            self.get_logger().info("Mode change: MONEY -> starting money pipeline.")
            starter = self._start_money_pipeline
        elif msg.mode == Mode.FACE:
            self.get_logger().info("Mode change: FACE -> starting face recognition pipeline.")
            starter = self._start_face_pipeline
        else:
            self.get_logger().info("Mode change: WALK -> starting depth pipeline.")
            starter = self._start_walk_pipeline
            self._last_prediction_text = "Walk mode"
        try:
            starter()
        except Exception as exc:
            self.get_logger().error(f"Failed to start {msg.mode} pipeline: {exc}")
            self._reconnect_due_ns = self.get_clock().now().nanoseconds + int(self._reconnect_period_sec * 1e9)
        self._publish_mode_heartbeat()

    def _poll_inference(self) -> None:
        """รอบทำงานหลัก (เรียกทุก 50 มิลลิวินาที): ดึงผลตรวจจับตามโหมดปัจจุบัน
        ถ้าไปป์ไลน์ยังไม่พร้อม/ผิดโหมด จะพยายามเริ่มใหม่ให้"""
        if self._current_mode == Mode.WALK:
            if self._publish_empty_navigation:
                self._navigation_publisher.publish(Detection2DArray())
            if not self._pipeline_ready or self._active_mode_pipeline != "walk":
                self._maybe_restart_pipeline()
                return
            self._poll_walk_inference()
            return
        # โหมดเงินและใบหน้าใช้ DetectionNetwork เหมือนกัน จึงใช้สาขาโค้ดเดียวกัน
        expected = "money" if self._current_mode == Mode.MONEY else "face"
        if not self._pipeline_ready or self._active_mode_pipeline != expected:
            self._maybe_restart_pipeline()
            return
        self._poll_det_preview()
        if self._det_queue is None:
            return

        try:
            # ดูดคิวให้หมด เก็บเฉพาะแพ็กเกตล่าสุด (เอาผลใหม่สุด ทิ้งของเก่าที่ตกค้าง)
            packet = self._det_queue.tryGet()
            while True:
                newer = self._det_queue.tryGet()
                if newer is None:
                    break
                packet = newer
        except Exception as exc:
            self._handle_pipeline_failure(f"{expected} detection queue closed: {exc}")
            return

        if packet is None:
            return

        # กรองด้วย threshold ต่อคลาสฝั่งโฮสต์ (NMS ทำบนตัวอุปกรณ์มาแล้ว)
        raw = list(getattr(packet, "detections", []) or [])
        kept = [d for d in raw if self._passes(d)]

        detections_msg = self._to_detection2d_array(kept)
        if kept:
            # ตัวที่มั่นใจสูงสุด ใช้เป็นข้อความสรุปและเก็บไว้แสดงผล
            top = max(kept, key=lambda d: d.confidence)
            label = self._archive_names.get(int(top.label), str(top.label))
            self._last_prediction_text = f"{label} ({top.confidence:.2f})"
            summary = ", ".join(
                f"id={d.label}({self._archive_names.get(int(d.label), '?')}) s={d.confidence:.2f}"
                for d in kept
            )
            self.get_logger().info(
                f"[{expected}] {len(kept)} -> {summary}",
                throttle_duration_sec=1.0,
            )

        if self._active_det_publisher is not None:
            self._active_det_publisher.publish(detections_msg)
        if self._publish_empty_navigation:
            self._navigation_publisher.publish(Detection2DArray())

    def _poll_walk_inference(self) -> None:
        """รอบทำงานโหมดเดิน: อ่าน depth -> หาสิ่งกีดขวางตามช่อง -> แล้วตรวจทางเดิน"""
        if self._walk_video_mode:
            self._poll_walk_video()
            return
        depth = None
        if self._walk_depth_queue is not None:
            try:
                dp = self._walk_depth_queue.tryGet()
                while True:
                    newer = self._walk_depth_queue.tryGet()
                    if newer is None:
                        break
                    dp = newer
                if dp is not None:
                    depth = dp.getCvFrame()
            except Exception as exc:
                self._handle_pipeline_failure(f"Walk depth queue closed: {exc}")
                return

        if depth is not None:
            zones = self._analyze_depth_zones(depth)
            obstacles = self._evaluate_obstacles(zones)
            if obstacles and self._tactile_obstacle_pause_sec > 0.0:
                # มีสิ่งกีดขวางใกล้ — พักการตรวจทางเดินไว้ก่อน เพื่อไม่ให้ขาของคน
                # ที่ยืนอยู่ตรงหน้าถูกเข้าใจผิดว่าเป็นทางเดิน
                self._tactile_pause_until_ns = (
                    self.get_clock().now().nanoseconds
                    + int(self._tactile_obstacle_pause_sec * 1e9)
                )
            self._publish_walk_detections(obstacles)
        self._poll_tactile_path()

    def _poll_walk_video(self) -> None:
        """รอบทำงานของโหมดวิดีโอชั่วคราว: อ่าน 1 เฟรม แล้วตรวจทางเดินบนเฟรมนั้น"""
        cap = self._walk_video_cap
        if cap is None:
            return
        import cv2

        ok, frame = cap.read()
        if not ok:                       # เล่นจบไฟล์แล้ว วนกลับไปต้นเรื่อง
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                return
        if frame.shape[1] != self._tactile_in_w or frame.shape[0] != self._tactile_in_h:
            frame = cv2.resize(frame, (self._tactile_in_w, self._tactile_in_h))
        self._run_tactile_on_frame(frame)

    def _poll_tactile_path(self) -> None:
        """ดึงเฟรม RGB ล่าสุดจากคิว tactile แล้วส่งไปตรวจหาทางเดิน"""
        if not self._tactile_enabled or self._tactile_queue is None:
            return
        try:
            frame_pkt = self._tactile_queue.tryGet()
            while True:
                newer = self._tactile_queue.tryGet()
                if newer is None:
                    break
                frame_pkt = newer
        except Exception as exc:
            self.get_logger().warning(
                f"Tactile queue closed: {exc}", throttle_duration_sec=2.0
            )
            self._tactile_queue = None
            return
        if frame_pkt is None:
            return
        try:
            bgr = frame_pkt.getCvFrame()
        except Exception:
            return
        self._run_tactile_on_frame(bgr)

    def _run_tactile_on_frame(self, bgr) -> None:
        # ผู้ใช้สั่ง "ปิดระบบนำทาง" — ไม่ตรวจ/ไม่ส่งทางเดิน (สิ่งกีดขวางยังทำงานปกติ)
        # แต่ยังส่งภาพดิบขึ้นพรีวิว เพื่อไม่ให้ภาพบนเว็บค้าง
        if not self._tactile_runtime_on:
            self._publish_preview(bgr)
            return
        # ระหว่างที่มีสิ่งกีดขวางใกล้ ให้ข้ามการตรวจทางเดินไปเลย (ขาของคนที่อยู่ใกล้
        # จะอ่านเป็นทางเดินผิด ๆ) แต่ยังส่งภาพดิบขึ้นพรีวิวบนเว็บอยู่ เพื่อไม่ให้ภาพค้าง
        if self.get_clock().now().nanoseconds < self._tactile_pause_until_ns:
            self._publish_preview(bgr)
            return
        try:
            _view, _mask, result = tactile_process(bgr, self._tactile_args)
        except Exception as exc:
            self.get_logger().warning(
                f"Tactile process failed: {exc}", throttle_duration_sec=2.0
            )
            return
        self._publish_tactile_result(result)
        # ส่งภาพที่ "วาดผลแล้ว" (ทางเดินทับสีเขียว, เส้นกลางสีน้ำเงิน, กากบาทจุดอ้างอิง
        # สีเหลือง + ข้อความทิศทาง) เป็นพรีวิวสดของโหมดเดิน เพื่อให้หน้าเว็บเห็นภาพ
        # แบบเดียวกับเครื่องมือทดสอบแยกต่างหาก ถ้าภาพนั้นใช้ไม่ได้ก็ถอยไปใช้ภาพดิบแทน
        self._publish_preview(_view if _view is not None else bgr)

    def _poll_det_preview(self) -> None:
        """ดึงเฟรมพรีวิวล่าสุดของโหมดเงิน/ใบหน้าแล้วส่งขึ้นหน้าเว็บ"""
        if self._det_preview_queue is None:
            return
        try:
            pkt = self._det_preview_queue.tryGet()
            while True:
                newer = self._det_preview_queue.tryGet()
                if newer is None:
                    break
                pkt = newer
        except Exception:
            self._det_preview_queue = None
            return
        if pkt is None:
            return
        try:
            bgr = pkt.getCvFrame()
        except Exception:
            return
        self._publish_preview(bgr)

    def _publish_preview(self, bgr) -> None:
        """ย่อภาพ -> เข้ารหัส JPEG -> ส่งขึ้น topic พรีวิวให้หน้าเว็บแสดง"""
        if self._preview_publisher is None or bgr is None:
            return
        try:
            import cv2
            h, w = bgr.shape[:2]
            if w > self._preview_max_width and w > 0:
                scale = self._preview_max_width / float(w)
                bgr = cv2.resize(bgr, (self._preview_max_width, max(1, int(h * scale))))
            ok, buf = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._preview_jpeg_quality]
            )
            if not ok:
                return
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self._preview_publisher.publish(msg)
        except Exception as exc:
            self.get_logger().warning(
                f"Preview publish failed: {exc}", throttle_duration_sec=5.0
            )

    def _publish_tactile_result(self, result) -> None:
        """แปลงผลตรวจทางเดิน (ทิศทาง/ระยะ/ทางแยก) เป็นข้อความ Detection2DArray แล้วส่งออก"""
        out = Detection2DArray()
        m = Detection2D()
        heading = result.get("heading") or ("NO PATH" if not result.get("found") else "")
        m.class_name = heading or "NO PATH"
        m.confidence = float(result.get("reach", 0.0))
        m.center_x = float(result.get("offset", 0.0))
        m.center_y = 1.0 if result.get("fork") else 0.0
        m.x_min = 0.0
        m.y_min = 0.0
        m.x_max = 1.0 if result.get("found") else 0.0
        m.y_max = 1.0 if result.get("found") else 0.0
        out.detections = [m]
        self._tactile_publisher.publish(out)
        if result.get("found"):
            self.get_logger().info(
                f"[tactile] {heading} off={result.get('offset', 0.0):+.2f} "
                f"reach={result.get('reach', 0.0):.0f}% fork={result.get('fork')}",
                throttle_duration_sec=1.0,
            )

    def _analyze_depth_zones(self, depth):
        """แบ่งภาพ depth เป็นช่องซ้าย/กลาง/ขวา แล้ววัดว่าในแต่ละช่องมีอะไรอยู่ใกล้แค่ไหน
        คืน dict 1 อันต่อช่อง บอกระยะที่ใกล้ที่สุด (มิลลิเมตร) และสัดส่วนพิกเซลที่อยู่ใกล้"""
        import numpy as np
        depth = np.asarray(depth)
        h, w = depth.shape[:2]
        top = max(0, min(h - 1, int(self._walk_roi_top_frac * h)))
        bottom = max(top + 1, min(h, int(self._walk_roi_bottom_frac * h)))
        roi = depth[top:bottom, :]
        rw = roi.shape[1]
        z_max = self._walk_obstacle_z_max_mm
        bounds = [
            ("left", 0, rw // 3),
            ("center", rw // 3, (2 * rw) // 3),
            ("right", (2 * rw) // 3, rw),
        ]
        zones = []
        for name, c0, c1 in bounds:
            sub = roi[:, c0:c1]
            area = int(sub.size)
            near = sub[(sub > 0) & (sub <= z_max)]
            frac = float(near.size) / area if area else 0.0
            # percentile ที่ 20 ของพิกเซลใกล้ ≈ ระยะใกล้ที่สุดที่ "นิ่งพอเชื่อได้"
            # ทนต่อพิกเซล depth ต่ำสุดที่เป็น noise หลงมาไม่กี่จุด
            z_near = float(np.percentile(near, 20)) if near.size else 0.0
            zones.append({
                "zone": name,
                "z_mm": z_near,
                "frac": frac,
                "col": (c0 / rw if rw else 0.0, c1 / rw if rw else 0.0),
                "row": (self._walk_roi_top_frac, self._walk_roi_bottom_frac),
            })
        return zones

    def _evaluate_obstacles(self, zones):
        """คัดเฉพาะช่องที่เข้าเกณฑ์เป็นสิ่งกีดขวาง (พิกเซลใกล้มากพอ และอยู่ในระยะ z_max)
        แล้ว log เตือน คืน list ของช่องที่เป็นสิ่งกีดขวาง"""
        obstacles = [
            z for z in zones
            if z["frac"] >= self._walk_obstacle_min_frac and 0 < z["z_mm"] <= self._walk_obstacle_z_max_mm
        ]
        if obstacles:
            summary = ", ".join(
                f"{z['zone']} {z['z_mm']/1000:.2f}m ({z['frac']*100:.0f}%)"
                for z in obstacles
            )
            self.get_logger().warning(
                f"[obstacle] {summary}",
                throttle_duration_sec=1.0,
            )
        return obstacles

    def _publish_walk_detections(self, obstacles) -> None:
        """ส่งสิ่งกีดขวางแต่ละช่องออก /navigation/detections
        (class_name = left/center/right, center_y = ระยะใกล้เป็นเมตร)"""
        out = Detection2DArray()
        msgs = []
        for z in obstacles:
            m = Detection2D()
            m.class_name = z["zone"]
            m.confidence = float(min(1.0, z["frac"]))
            c0, c1 = z["col"]
            r0, r1 = z["row"]
            m.x_min = float(c0)
            m.y_min = float(r0)
            m.x_max = float(c1)
            m.y_max = float(r1)
            m.center_x = float((c0 + c1) * 0.5)
            m.center_y = float(z["z_mm"]) / 1000.0
            msgs.append(m)
        out.detections = msgs
        self._navigation_publisher.publish(out)

    def _passes(self, d) -> bool:
        """ผ่านเกณฑ์ไหม: ความมั่นใจของ detection ต้อง >= threshold ของคลาสนั้น"""
        idx = int(d.label)
        if idx < 0 or idx >= len(self._active_thresholds):
            return False
        return float(d.confidence) >= self._active_thresholds[idx]

    def _to_detection2d_array(self, dets):
        """แปลง detection ของ DepthAI เป็นข้อความ ROS Detection2DArray (ใส่ชื่อคลาส+กรอบ)"""
        out = Detection2DArray()
        msgs = []
        for d in dets:
            idx = int(d.label)
            label = self._archive_names.get(idx, str(idx))
            m = Detection2D()
            m.class_name = label
            m.confidence = float(d.confidence)
            m.x_min = float(d.xmin)
            m.y_min = float(d.ymin)
            m.x_max = float(d.xmax)
            m.y_max = float(d.ymax)
            m.center_x = float((d.xmin + d.xmax) * 0.5)
            m.center_y = float((d.ymin + d.ymax) * 0.5)
            msgs.append(m)
        out.detections = msgs
        return out

    def _handle_pipeline_failure(self, message: str) -> None:
        """จัดการเมื่อไปป์ไลน์/อุปกรณ์ล้มเหลว: ปิดทิ้ง แล้วตั้งเวลาลองเชื่อมใหม่"""
        self.get_logger().error(message)
        self._cleanup_pipeline()
        self._reconnect_due_ns = self.get_clock().now().nanoseconds + int(self._reconnect_period_sec * 1e9)
        self._last_prediction_text = "Pipeline reconnecting"

    def _maybe_restart_pipeline(self) -> None:
        """ถ้าถึงเวลาที่ตั้งไว้ ให้ลองเปิดไปป์ไลน์ของโหมดปัจจุบันใหม่อีกครั้ง"""
        if self._reconnect_due_ns == 0:
            return
        if self.get_clock().now().nanoseconds < self._reconnect_due_ns:
            return
        if self._current_mode == Mode.MONEY:
            target, starter = "money", self._start_money_pipeline
        elif self._current_mode == Mode.FACE:
            target, starter = "face", self._start_face_pipeline
        else:
            target, starter = "walk", self._start_walk_pipeline
        self.get_logger().warning(f"Restarting {target} pipeline after device/queue failure.")
        try:
            starter()
        except Exception as exc:
            self.get_logger().error(f"{target} pipeline restart failed: {exc}")
            self._reconnect_due_ns = self.get_clock().now().nanoseconds + int(self._reconnect_period_sec * 1e9)

    def _cleanup_pipeline(self) -> None:
        """ปิดไปป์ไลน์ปัจจุบันและล้างคิว/อุปกรณ์ทั้งหมด (เรียกก่อนสลับโหมดหรือตอนปิดโปรแกรม)"""
        self._pipeline_ready = False
        self._active_mode_pipeline = None
        self._det_queue = None
        self._walk_depth_queue = None
        self._tactile_queue = None
        self._det_preview_queue = None
        if getattr(self, "_walk_video_cap", None) is not None:
            try:
                self._walk_video_cap.release()
            except Exception:
                pass
        self._walk_video_cap = None
        self._walk_video_mode = False
        self._active_det_publisher = None
        self._pipeline = None
        if getattr(self, "_device", None) is not None:
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None


def main(args=None) -> None:
    """จุดเริ่มโปรแกรม: สร้าง node, รันวนรับ event, และเก็บกวาดให้เรียบร้อยตอนปิด"""
    rclpy.init(args=args)
    node = MainPipelineNode()
    try:
        rclpy.spin(node)
    finally:
        node._cleanup_pipeline()
        node.destroy_node()
        rclpy.shutdown()
