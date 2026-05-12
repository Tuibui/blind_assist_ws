import queue
import shutil
import subprocess
import threading
import time
from collections import Counter

import rclpy
from rclpy.node import Node

from oak_interfaces.msg import Detection2DArray, Mode


class DecisionAudioNode(Node):
    def __init__(self) -> None:
        super().__init__("decision_audio_node")

        self.declare_parameter("money_detections_topic", "/money/detections")
        self.declare_parameter("navigation_detections_topic", "/navigation/detections")
        self.declare_parameter("current_mode_topic", "/oak/current_mode")
        self.declare_parameter("cooldown_sec", 2.0)
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("tts_command", "edge-playback")
        self.declare_parameter("tts_voice_th", "th-TH-PremwadeeNeural")
        self.declare_parameter("tts_voice_en", "en-US-EmmaNeural")
        self.declare_parameter("speak_zero_detections", False)
        self.declare_parameter("announce_mode_changes", True)
        self.declare_parameter("announce_detections", False)

        money_topic = self.get_parameter("money_detections_topic").value
        navigation_topic = self.get_parameter("navigation_detections_topic").value
        current_mode_topic = self.get_parameter("current_mode_topic").value
        self._cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self._tts_enabled = bool(self.get_parameter("tts_enabled").value)
        self._tts_command = str(self.get_parameter("tts_command").value).strip()
        self._tts_voice_th = str(self.get_parameter("tts_voice_th").value).strip()
        self._tts_voice_en = str(self.get_parameter("tts_voice_en").value).strip()
        self._speak_zero_detections = bool(self.get_parameter("speak_zero_detections").value)
        self._announce_mode_changes = bool(self.get_parameter("announce_mode_changes").value)
        self._announce_detections = bool(self.get_parameter("announce_detections").value)
        self._last_log_time = {}
        self._last_summary = {}
        self._last_mode = None
        self._speech_queue = queue.Queue()
        self._running = True
        self._tts_available = bool(self._tts_command) and shutil.which(self._tts_command) is not None

        if self._tts_enabled and not self._tts_available:
            self.get_logger().warning(
                f'TTS command "{self._tts_command}" is not installed. Decision audio will log only.'
            )

        self._speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._speech_thread.start()

        self.create_subscription(Detection2DArray, money_topic, self._on_money_detections, 10)
        self.create_subscription(Detection2DArray, navigation_topic, self._on_navigation_detections, 10)
        self.create_subscription(Mode, current_mode_topic, self._on_mode, 10)

    def _should_log(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last_log_time.get(key, 0.0)
        if now - last < self._cooldown_sec:
            return False
        self._last_log_time[key] = now
        return True

    def _speech_loop(self) -> None:
        while self._running:
            try:
                utterance = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if utterance is None:
                return
            self._speak_item(utterance)

    def _announce_bilingual(self, text_th: str, text_en: str) -> None:
        self._enqueue_bilingual_speech(text_th, text_en)

    def _speak_item(self, item) -> None:
        if isinstance(item, list):
            for text, voice in item:
                self._speak_text(text, voice)
            return
        self._speak_text(item, self._tts_voice_th)

    def _speak_text(self, text: str, voice: str) -> None:
        if not self._tts_enabled or not self._tts_available:
            return
        cmd = [self._tts_command]
        if self._tts_command == "edge-playback" and voice:
            cmd.extend(["--voice", voice, "--text"])
        elif self._tts_command in ("espeak", "espeak-ng") and voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as exc:
            self.get_logger().warning(f"Failed to speak audio prompt: {exc}")

    def _enqueue_speech(self, text: str) -> None:
        if not text:
            return
        try:
            while self._speech_queue.qsize() > 1:
                self._speech_queue.get_nowait()
        except queue.Empty:
            pass
        self._speech_queue.put_nowait(text)

    def _enqueue_bilingual_speech(self, text_th: str, text_en: str) -> None:
        segments = []
        if text_th:
            segments.append((text_th, self._tts_voice_th))
        if text_en:
            segments.append((text_en, self._tts_voice_en))
        if not segments:
            return
        try:
            while self._speech_queue.qsize() > 1:
                self._speech_queue.get_nowait()
        except queue.Empty:
            pass
        self._speech_queue.put_nowait(segments)

    def _build_detection_summary(self, label: str, msg: Detection2DArray) -> tuple[str, str]:
        count = len(msg.detections)
        if count == 0:
            summary = f"{label} detections=0"
            speech = f"No {label} detections"
            return summary, speech

        class_counts = Counter(
            detection.class_name.strip() or label
            for detection in msg.detections
        )
        parts = []
        for class_name, class_count in class_counts.items():
            if class_count == 1:
                parts.append(class_name)
            else:
                parts.append(f"{class_count} {class_name}")
        joined = ", ".join(parts)
        summary = f"{label} detections={count} [{joined}]"
        speech = f"{label}: {joined}"
        return summary, speech

    def _handle_detections(self, key: str, label: str, msg: Detection2DArray) -> None:
        if not self._announce_detections:
            return
        summary, speech = self._build_detection_summary(label, msg)
        count = len(msg.detections)
        if count == 0 and self._last_summary.get(key) == summary:
            return
        if self._should_log(key):
            self._last_summary[key] = summary
            self.get_logger().info(f"Decision audio: {summary}")
            if len(msg.detections) > 0 or self._speak_zero_detections:
                self._enqueue_speech(speech)

    def _on_money_detections(self, msg: Detection2DArray) -> None:
        self._handle_detections("money", "money", msg)

    def _on_navigation_detections(self, msg: Detection2DArray) -> None:
        self._handle_detections("navigation", "navigation", msg)

    def _on_mode(self, msg: Mode) -> None:
        if not self._announce_mode_changes:
            return
        if msg.mode not in (Mode.WALK, Mode.MONEY):
            return
        if msg.mode == self._last_mode:
            return
        self._last_mode = msg.mode
        mode_name = "WALK" if msg.mode == Mode.WALK else "MONEY"
        speech_th = "โหมดเดิน" if msg.mode == Mode.WALK else "โหมดเงิน"
        speech_en = "Walk mode" if msg.mode == Mode.WALK else "Money mode"
        self.get_logger().info(f"Decision audio: current mode is {mode_name} ({msg.mode})")
        self._announce_bilingual(speech_th, speech_en)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionAudioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        if hasattr(node, "_speech_queue"):
            node._speech_queue.put_nowait(None)
        if hasattr(node, "_speech_thread"):
            node._speech_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
