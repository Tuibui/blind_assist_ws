"""node ตัดสินใจและออกเสียง — "สมอง" ฝั่งเสียงของระบบ

รับผลตรวจจับจาก main_pipeline (เงิน / สิ่งกีดขวาง / ใบหน้า / ทางเดิน) แล้วตัดสินใจว่า
ควรพูดอะไรกับผู้ใช้ พร้อมจัดลำดับความสำคัญของเสียงไม่ให้พูดทับกัน และส่งข้อความ
ที่กำลังพูดขึ้นหน้าเว็บด้วย (/oak/announcement)

หัวใจสำคัญคือ "ตัวจัดการเสียง" (speech manager) ที่อธิบายไว้เหนือเมธอด _say()
"""

import hashlib
import os
import shutil
import subprocess
import threading
import time
from collections import Counter, deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from oak_interfaces.msg import Detection2DArray, Mode

from .audio_env import ensure_audio_session_env


# ตารางแปลงชื่อคลาสธนบัตร -> ประโยคพูด (ไทย, อังกฤษ)
# หมายเหตุ: key ที่นี่เป็นรูปแบบเก่า ("0_20", ...) แต่ main_pipeline ปัจจุบันส่ง
# class_name มาเป็น "20"/"50"/"100"/"500"/"1000" จึงไม่ตรงกัน ทำให้ระบบพูดเป็น
# ตัวเลขดิบ (เช่น "20") แทน "20 บาท" — ควรแก้ key ให้ตรงกับ money_labels ใน
# main_pipeline.yaml (ดูสรุปประเด็นท้ายงาน)
MONEY_LABEL_SPEECH = {
    "0_20": ("20 บาท", "20 baht"),
    "1_50": ("50 บาท", "50 baht"),
    "2_100": ("100 บาท", "100 baht"),
    "3_500": ("500 บาท", "500 baht"),
    "4_1000": ("1000 บาท", "1000 baht"),
}

# คำบอกทิศทางเดินตามทางเดิน (tactile) ตัวตรวจจะส่งทิศมาเป็นข้อความ เช่น "STRAIGHT",
# "BEAR LEFT 14%", "SHARP RIGHT 32%", "STOP - DECIDE" เรายุบให้เหลือ 4 หมวดพูด
# (ทั้ง BEAR และ SHARP กลายเป็น "เลี้ยวซ้าย/ขวา" ธรรมดา)
TACTILE_SPEECH = {
    "STRAIGHT": ("ตรงไป", "Go straight"),
    "LEFT": ("เลี้ยวซ้าย", "Turn left"),
    "RIGHT": ("เลี้ยวขวา", "Turn right"),
    "STOP": ("หยุด ตัดสินใจ", "Stop and decide"),
}

# ระดับความสำคัญของเสียง — ตัวเลขสูงกว่าจะ "แทรก" (ตัด) ตัวที่กำลังพูดอยู่ที่ต่ำกว่า
# และถูกหยิบไปพูดก่อนเมื่อมีหลายอันรอคิว
PRIO_DIR = 10       # บอกทิศทางเดิน (ตรงไป / ซ้าย / ขวา)
PRIO_INFO = 20      # ทักทายตอนเปิดเครื่อง / สถานะ
PRIO_MODE = 30      # แจ้งเปลี่ยนโหมด
PRIO_RESULT = 40    # ผลธนบัตร / ใบหน้า (ผู้ใช้เป็นคนขอ)
PRIO_ALERT = 50     # สิ่งกีดขวาง / ให้หยุดตัดสินใจ (ความปลอดภัย)
PRIO_POWER = 60     # ไฟเลี้ยงต่ำ (ความปลอดภัยของระบบ)


class _Utterance:
    """หนึ่ง "ประโยคที่จะพูด" พร้อม metadata ที่ตัวจัดการเสียงใช้ตัดสินลำดับ"""

    __slots__ = ("text", "voice", "priority", "key", "created_at", "expires_at")

    def __init__(self, text, voice, priority, key, created_at, expires_at):
        self.text = text
        self.voice = voice
        self.priority = priority
        self.key = key
        self.created_at = created_at
        self.expires_at = expires_at


class DecisionAudioNode(Node):
    """node ที่ตัดสินใจว่าจะพูดอะไร และออกเสียงให้ผู้ใช้ (พร้อมส่งข้อความขึ้นหน้าเว็บ)"""

    def __init__(self) -> None:
        super().__init__("decision_audio_node")

        # ---- ประกาศพารามิเตอร์ทั้งหมด (ปรับค่าได้ผ่านไฟล์ oak_decision_audio.yaml) ----
        self.declare_parameter("money_detections_topic", "/money/detections")
        self.declare_parameter("navigation_detections_topic", "/navigation/detections")
        self.declare_parameter("face_detections_topic", "/face/detections")
        self.declare_parameter("tactile_detections_topic", "/tactile/path")
        # การบอกทิศทางเดินตามทางเดิน (tactile) จะพูดก็ต่อเมื่อ "หมวดทิศเปลี่ยน" และ
        # นิ่งมานานพอเท่านั้น (topic นี้มาที่ ~20 Hz เราจึงห้ามพูดทุกเฟรม)
        # การเตือนสิ่งกีดขวางมีความสำคัญกว่า จะ "ปิดปาก" การบอกทิศนี้ชั่วคราว
        # นาน tactile_obstacle_mute_sec
        self.declare_parameter("tactile_announce_enabled", True)
        # ทิศทางตัดสินจาก "เสียงข้างมาก" ของเฟรมในช่วง tactile_window_sec ล่าสุด
        # (ทนต่อการกระพริบสลับ STRAIGHT/LEFT/RIGHT รายเฟรม) และต้องมีอย่างน้อย
        # tactile_min_samples ตัวอย่างในหน้าต่าง พอทิศชนะเปลี่ยนก็พูดทันที
        self.declare_parameter("tactile_window_sec", 1.0)
        self.declare_parameter("tactile_min_samples", 5)
        self.declare_parameter("tactile_obstacle_mute_sec", 2.0)
        # กันพูดกลับไปกลับมา: หลังพูดทิศหนึ่งแล้ว ต้องค้างทิศนั้นอย่างน้อยเท่านี้ก่อน
        # จะประกาศทิศ "อื่น" เพื่อไม่ให้ทางที่ส่าย ๆ ทำให้พูดรัว (ยกเว้น STOP/ทางแยก
        # ซึ่งเร่งด่วน จะข้ามกฎนี้)
        self.declare_parameter("tactile_min_dwell_sec", 2.5)
        # เวลาขั้นต่ำก่อนพูดชื่อใบหน้า "เดิม" ซ้ำ (ถ้าเป็นชื่อใหม่จะพูดทันที)
        self.declare_parameter("face_repeat_sec", 5.0)
        self.declare_parameter("current_mode_topic", "/oak/current_mode")
        self.declare_parameter("cooldown_sec", 2.0)
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("tts_command", "edge-playback")
        self.declare_parameter("tts_voice_th", "th-TH-PremwadeeNeural")
        self.declare_parameter("tts_voice_en", "en-US-EmmaNeural")
        self.declare_parameter("announce_mode_changes", True)
        self.declare_parameter("announce_detections", False)
        self.declare_parameter("boot_greeting_enabled", True)
        self.declare_parameter("boot_greeting_th", "ระบบพร้อมใช้งาน โหมดเดิน")
        self.declare_parameter("boot_greeting_en", "System ready. Walk mode.")
        self.declare_parameter("money_confirm_count", 5)
        self.declare_parameter("money_gap_sec", 3.0)
        self.declare_parameter("language_topic", "/oak/language")
        self.declare_parameter("default_language", "th")
        # ข้อความที่แสดงบนหน้าเว็บ (สิ่งล่าสุดที่ประกาศให้ผู้ใช้)
        self.declare_parameter("announcement_topic", "/oak/announcement")
        # เฝ้าระวังไฟเลี้ยงต่ำของ Raspberry Pi (ผ่านคำสั่ง `vcgencmd get_throttled`)
        self.declare_parameter("undervoltage_check_enabled", True)
        self.declare_parameter("undervoltage_check_interval_sec", 10.0)
        self.declare_parameter("undervoltage_repeat_sec", 60.0)
        self.declare_parameter(
            "undervoltage_th", "คำเตือน ไฟเลี้ยงต่ำ กรุณาตรวจสอบแบตเตอรี่"
        )
        self.declare_parameter(
            "undervoltage_en", "Warning, low power supply. Please check the battery."
        )

        money_topic = self.get_parameter("money_detections_topic").value
        navigation_topic = self.get_parameter("navigation_detections_topic").value
        face_topic = self.get_parameter("face_detections_topic").value
        tactile_topic = self.get_parameter("tactile_detections_topic").value
        self._tactile_announce_enabled = bool(self.get_parameter("tactile_announce_enabled").value)
        self._tactile_window_sec = float(self.get_parameter("tactile_window_sec").value)
        self._tactile_min_samples = max(1, int(self.get_parameter("tactile_min_samples").value))
        self._tactile_obstacle_mute_sec = float(self.get_parameter("tactile_obstacle_mute_sec").value)
        self._tactile_min_dwell_sec = float(self.get_parameter("tactile_min_dwell_sec").value)
        self._face_repeat_sec = float(self.get_parameter("face_repeat_sec").value)
        current_mode_topic = self.get_parameter("current_mode_topic").value
        self._cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self._tts_enabled = bool(self.get_parameter("tts_enabled").value)
        self._tts_command = str(self.get_parameter("tts_command").value).strip()
        self._tts_voice_th = str(self.get_parameter("tts_voice_th").value).strip()
        self._tts_voice_en = str(self.get_parameter("tts_voice_en").value).strip()
        self._announce_mode_changes = bool(self.get_parameter("announce_mode_changes").value)
        self._announce_detections = bool(self.get_parameter("announce_detections").value)
        self._money_confirm_count = max(1, int(self.get_parameter("money_confirm_count").value))
        self._money_gap_sec = float(self.get_parameter("money_gap_sec").value)
        language_topic = str(self.get_parameter("language_topic").value)
        default_lang = str(self.get_parameter("default_language").value).strip().lower()
        self._language = "en" if default_lang.startswith("en") else "th"
        self._last_log_time = {}
        self._last_summary = {}
        self._last_mode = None
        self._money_window = []
        self._money_gap_until = 0.0
        self._last_face_name = None
        self._face_announce_until = 0.0
        # สถานะการบอกทิศทางเดิน (โหมดเดิน)
        self._obstacle_active_until = 0.0       # ปิดปากการบอกทิศจนกว่าเวลาจะเลยค่านี้
        self._tactile_last_announced = None     # หมวดทิศที่พูดไปล่าสุด
        self._tactile_last_change_at = 0.0      # เวลาที่เปลี่ยนทิศครั้งล่าสุด
        self._tactile_window = deque()          # หน้าต่างเลื่อน (เวลา, หมวดทิศ)
        self._running = True

        # ---- สถานะของตัวจัดการเสียง (ดูคำอธิบายดีไซน์ที่เมธอดด้านล่าง) ----
        self._speech_cv = threading.Condition(threading.Lock())
        self._pending = {}              # key -> _Utterance (เก็บอันล่าสุดอันเดียวต่อ key)
        self._current = None            # _Utterance ที่กำลังพูดอยู่ตอนนี้
        self._current_proc = None       # subprocess เล่นเสียงของมัน (ไว้สั่งตัดเพื่อแทรก)
        self._last_spoken_at = {}       # key -> เวลาที่พูดจบล่าสุด
        self._last_spoken_text = {}     # key -> ข้อความที่พูดล่าสุด (ไว้กันพูดซ้ำ)
        self._inter_gap = 0.12          # เว้นเงียบสั้น ๆ ระหว่างแต่ละประโยค

        # แบ็กเอนด์เสียง วิธีที่อยากใช้: edge-tts สังเคราะห์ประโยคเป็นไฟล์ mp3 แล้ว
        # cache ไว้ครั้งเดียว จากนั้นตัวเล่นในเครื่อง (mpv/ffplay) เล่นได้ทันทีและออฟไลน์
        # ทางสำรอง: คำสั่งสังเคราะห์-แล้วเล่นแบบเดิม (ช้ากว่า และต้องต่อเน็ต)
        self._tts_cache_dir = Path(
            os.environ.get("BLIND_ASSIST_TTS_CACHE", "/tmp/blind_assist_tts")
        )
        try:
            self._tts_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if shutil.which("mpv"):
            self._player_cmd = ["mpv", "--no-video", "--really-quiet", "--no-terminal"]
        elif shutil.which("ffplay"):
            self._player_cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        else:
            self._player_cmd = None
        self._can_cache = bool(shutil.which("edge-tts")) and self._player_cmd is not None
        self._has_fallback = bool(self._tts_command) and shutil.which(self._tts_command) is not None
        self._tts_active = self._tts_enabled and (self._can_cache or self._has_fallback)

        if not self._tts_active and self._tts_enabled:
            self.get_logger().warning(
                "No TTS backend found (need edge-tts + mpv/ffplay, or "
                f'"{self._tts_command}"). Decision audio will log only.'
            )
        elif self._tts_active and not self._can_cache:
            self.get_logger().warning(
                "edge-tts/mpv not found; using slower un-cached TTS fallback."
            )

        self._speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._speech_thread.start()
        if self._tts_active and self._can_cache:
            # สังเคราะห์ประโยคคงที่ล่วงหน้าในเบื้องหลัง เพื่อให้ครั้งแรกที่ต้องใช้แต่ละ
            # ประโยคเล่นได้ทันที (และหลังจากนั้นใช้ได้แบบออฟไลน์)
            prewarm = (
                list(TACTILE_SPEECH.values())
                + list(MONEY_LABEL_SPEECH.values())
                + [
                    ("ระวัง สิ่งกีดขวาง", "Obstacle ahead"),
                    ("โหมดเดิน", "Walk mode"),
                    ("โหมดเงิน", "Money mode"),
                    (
                        str(self.get_parameter("boot_greeting_th").value),
                        str(self.get_parameter("boot_greeting_en").value),
                    ),
                    (
                        str(self.get_parameter("undervoltage_th").value),
                        str(self.get_parameter("undervoltage_en").value),
                    ),
                ]
            )
            threading.Thread(target=self._prewarm, args=(prewarm,), daemon=True).start()

        self._announcement_pub = self.create_publisher(
            String, str(self.get_parameter("announcement_topic").value), 10
        )

        # ---- รับผลตรวจจับทุกประเภทจาก main_pipeline + สถานะโหมด/ภาษา ----
        self.create_subscription(Detection2DArray, money_topic, self._on_money_detections, 10)
        self.create_subscription(Detection2DArray, navigation_topic, self._on_navigation_detections, 10)
        self.create_subscription(Detection2DArray, face_topic, self._on_face_detections, 10)
        self.create_subscription(Detection2DArray, tactile_topic, self._on_tactile_path, 10)
        self.create_subscription(Mode, current_mode_topic, self._on_mode, 10)
        self.create_subscription(String, language_topic, self._on_language, 10)

        self._undervoltage_repeat_sec = float(
            self.get_parameter("undervoltage_repeat_sec").value
        )
        self._undervoltage_warned_at = 0.0
        self._undervoltage_timer = None
        self._undervoltage_available = shutil.which("vcgencmd") is not None
        if bool(self.get_parameter("undervoltage_check_enabled").value):
            interval = max(1.0, float(self.get_parameter("undervoltage_check_interval_sec").value))
            if self._undervoltage_available:
                self._undervoltage_timer = self.create_timer(interval, self._check_undervoltage)
            else:
                self.get_logger().warning(
                    "vcgencmd not found; Raspberry Pi under-voltage monitoring disabled."
                )

        # ทักทายตอนเปิดเครื่อง บอกว่าระบบพร้อมและอยู่โหมดเดิน
        if bool(self.get_parameter("boot_greeting_enabled").value):
            greeting_th = str(self.get_parameter("boot_greeting_th").value)
            greeting_en = str(self.get_parameter("boot_greeting_en").value)
            self.get_logger().info(f"Boot greeting: {greeting_th} / {greeting_en}")
            self._say(greeting_th, greeting_en, key="status", priority=PRIO_INFO)

    def _should_log(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last_log_time.get(key, 0.0)
        if now - last < self._cooldown_sec:
            return False
        self._last_log_time[key] = now
        return True

    # ----------------------------- Speech manager -----------------------------
    # One worker thread plays at most ONE utterance at a time, so speech can
    # never overlap. Callers hand work to _say(); the manager then:
    #   * coalesces — keeps only the latest utterance of each `key`, so a burst
    #     of directions never piles into a backlog;
    #   * prioritises — a safety alert preempts (kills) a direction mid-sentence
    #     and is spoken first;
    #   * dedups — the same phrase isn't repeated within `min_repeat` seconds;
    #   * expires — an utterance older than its `ttl` is dropped unspoken (a
    #     turn you already passed is never announced late).

    def _say(self, text_th, text_en, *, key, priority, min_repeat=0.0, ttl=None):
        """Queue a bilingual announcement; the active language is resolved now."""
        text = text_en if self._language == "en" else text_th
        voice = self._tts_voice_en if self._language == "en" else self._tts_voice_th
        if not text:
            return
        now = time.monotonic()
        with self._speech_cv:
            if (
                min_repeat > 0.0
                and self._last_spoken_text.get(key) == text
                and now - self._last_spoken_at.get(key, -1e9) < min_repeat
            ):
                return  # same thing said too recently — skip it
            self._pending[key] = _Utterance(
                text, voice, priority, key, now, (now + ttl) if ttl else None
            )
            # Preempt a lower-priority utterance that is currently playing.
            if (
                self._current is not None
                and priority > self._current.priority
                and self._current_proc is not None
            ):
                self._terminate_proc(self._current_proc)
            self._speech_cv.notify()
        self._publish_announcement(text)

    def _publish_announcement(self, text: str) -> None:
        """Push the latest spoken phrase to the web dashboard."""
        if not text:
            return
        msg = String()
        msg.data = text
        self._announcement_pub.publish(msg)

    def _pop_best_locked(self, now):
        """Drop expired pending utterances, return the most important remaining
        one (highest priority, newest on a tie) and remove it. Caller holds CV."""
        dead = [
            k for k, u in self._pending.items()
            if u.expires_at is not None and u.expires_at < now
        ]
        for k in dead:
            del self._pending[k]
        if not self._pending:
            return None
        key = max(
            self._pending,
            key=lambda k: (self._pending[k].priority, self._pending[k].created_at),
        )
        return self._pending.pop(key)

    def _speech_loop(self) -> None:
        while self._running:
            with self._speech_cv:
                utt = self._pop_best_locked(time.monotonic())
                while utt is None and self._running:
                    self._speech_cv.wait(timeout=0.2)
                    utt = self._pop_best_locked(time.monotonic())
                if not self._running:
                    return
                self._current = utt
            self._play(utt)
            with self._speech_cv:
                self._last_spoken_at[utt.key] = time.monotonic()
                self._last_spoken_text[utt.key] = utt.text
                self._current = None
                self._current_proc = None
            time.sleep(self._inter_gap)  # brief gap so words don't run together

    @staticmethod
    def _terminate_proc(proc) -> None:
        try:
            proc.terminate()
        except Exception:
            pass

    def _play(self, utt) -> None:
        if not self._tts_active:
            return
        path = self._audio_path(utt.text, utt.voice)
        if path is not None and self._player_cmd:
            cmd = self._player_cmd + [str(path)]
        elif self._has_fallback:
            cmd = self._build_fallback_cmd(utt.text, utt.voice)
        else:
            return
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as exc:
            self.get_logger().warning(f"Failed to start audio playback: {exc}")
            return
        with self._speech_cv:
            self._current_proc = proc
        try:
            proc.wait()
        except Exception:
            pass

    def _audio_path(self, text, voice):
        """Return a cached audio file for (text, voice), synthesising on a miss.
        Returns None if caching isn't possible (no edge-tts, or offline miss)."""
        if not self._can_cache:
            return None
        h = hashlib.md5(f"{voice}|{text}".encode("utf-8")).hexdigest()
        path = self._tts_cache_dir / f"{h}.mp3"
        if path.exists() and path.stat().st_size > 0:
            return path
        try:
            subprocess.run(
                ["edge-tts", "--voice", voice, "--text", text,
                 "--write-media", str(path)],
                check=True, capture_output=True, timeout=20.0,
            )
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        return path if path.exists() and path.stat().st_size > 0 else None

    def _build_fallback_cmd(self, text, voice):
        cmd = [self._tts_command]
        if self._tts_command == "edge-playback" and voice:
            cmd.extend(["--voice", voice, "--text"])
        elif self._tts_command in ("espeak", "espeak-ng") and voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
        return cmd

    def _prewarm(self, phrases) -> None:
        for text_th, text_en in phrases:
            if not self._running:
                return
            text = text_en if self._language == "en" else text_th
            voice = self._tts_voice_en if self._language == "en" else self._tts_voice_th
            if text:
                self._audio_path(text, voice)

    def _on_language(self, msg: String) -> None:
        new_lang = (msg.data or "").strip().lower()
        if new_lang not in ("th", "en"):
            return
        if new_lang == self._language:
            return
        self._language = new_lang
        self.get_logger().info(f"Language switched to {new_lang}")

    def _on_money_detections(self, msg: Detection2DArray) -> None:
        if not self._announce_detections:
            return
        now = time.monotonic()

        if now < self._money_gap_until:
            self._money_window.clear()
            return

        if not msg.detections:
            # Tolerate dropped/empty frames: skip this frame without resetting
            # the vote window, so a single missed detection doesn't restart the
            # majority vote.
            return

        best = max(msg.detections, key=lambda d: float(d.confidence))
        label = best.class_name

        # Collect a window of recent detections, then announce the most
        # frequently seen label (majority vote) instead of requiring the
        # same label on consecutive frames.
        self._money_window.append(label)
        self.get_logger().info(
            f"Money sample {len(self._money_window)}/{self._money_confirm_count}: "
            f"{label} ({best.confidence:.2f})"
        )
        if len(self._money_window) < self._money_confirm_count:
            return

        winner, votes = Counter(self._money_window).most_common(1)[0]
        speech_th, speech_en = MONEY_LABEL_SPEECH.get(winner, (winner, winner))
        self.get_logger().info(
            f"Money confirmed: {winner} with {votes}/{len(self._money_window)} votes; "
            f"gap {self._money_gap_sec:.1f}s"
        )
        self._say(speech_th, speech_en, key="money", priority=PRIO_RESULT)
        self._money_window.clear()
        self._money_gap_until = now + self._money_gap_sec

    def _on_navigation_detections(self, msg: Detection2DArray) -> None:
        # Any near obstacle in any zone -> one plain warning, no direction.
        has_obstacle = any(d.class_name.strip() for d in msg.detections)
        # Obstacles take priority over tactile direction guidance: while one is
        # present (and shortly after), mute the spoken directions. This runs
        # regardless of announce_detections so the priority is always honoured.
        if has_obstacle:
            self._obstacle_active_until = time.monotonic() + self._tactile_obstacle_mute_sec
        if not self._announce_detections:
            return
        if not has_obstacle:
            return
        # cooldown_sec gate keeps this from spamming every frame
        if not self._should_log("navigation"):
            return
        self.get_logger().warning("Obstacle ahead")
        self._say("ระวัง สิ่งกีดขวาง", "Obstacle ahead",
                  key="alert", priority=PRIO_ALERT, ttl=1.5)

    @staticmethod
    def _tactile_category(heading: str):
        """Map a raw tactile heading string to a spoken category (or None)."""
        h = (heading or "").strip().upper()
        if not h or h.startswith("??") or h == "NO PATH":
            return None
        if h.startswith("STOP"):
            return "STOP"
        if h.startswith("STRAIGHT"):
            return "STRAIGHT"
        if "LEFT" in h:                          # BEAR LEFT / SHARP LEFT
            return "LEFT"
        if "RIGHT" in h:                         # BEAR RIGHT / SHARP RIGHT
            return "RIGHT"
        return None

    def _on_tactile_path(self, msg: Detection2DArray) -> None:
        """Speak walk-mode path direction, announcing only when it changes.

        The detector publishes a heading every frame (~20 Hz) as the first
        detection's class_name, and it flickers (STRAIGHT<->LEFT<->RIGHT). So we
        keep a sliding window of the last tactile_window_sec of frames and speak
        the *majority* direction — this catches a real turn even when individual
        frames jitter, instead of demanding an unbroken run of identical frames.
        The user hears "เลี้ยวซ้าย"/"ตรงไป"/"หยุด ตัดสินใจ" when the dominant
        direction actually changes.
        """
        if not self._tactile_announce_enabled:
            return
        heading = msg.detections[0].class_name if msg.detections else ""
        cat = self._tactile_category(heading)
        now = time.monotonic()

        # Maintain the sliding window (drop entries older than the window).
        if cat is not None:
            self._tactile_window.append((now, cat))
        cutoff = now - self._tactile_window_sec
        while self._tactile_window and self._tactile_window[0][0] < cutoff:
            self._tactile_window.popleft()

        if now < self._obstacle_active_until:    # obstacle warning has priority
            return
        if len(self._tactile_window) < self._tactile_min_samples:
            return                               # not enough evidence yet

        winner, votes = Counter(c for _, c in self._tactile_window).most_common(1)[0]
        if winner == self._tactile_last_announced:
            return                               # already told them this
        # Anti-flip-flop: once a direction is spoken, hold it for
        # tactile_min_dwell_sec before switching to a different one — a STOP
        # (junction) is urgent and bypasses the hold.
        if winner != "STOP" and (now - self._tactile_last_change_at) < self._tactile_min_dwell_sec:
            return
        self._tactile_last_announced = winner
        self._tactile_last_change_at = now
        speech_th, speech_en = TACTILE_SPEECH[winner]
        prio = PRIO_ALERT if winner == "STOP" else PRIO_DIR
        self.get_logger().info(
            f"[nav] tactile -> {winner} ({votes}/{len(self._tactile_window)} votes): {speech_th}"
        )
        self._say(speech_th, speech_en, key="dir", priority=prio,
                  ttl=None if winner == "STOP" else 1.5)

    def _on_face_detections(self, msg: Detection2DArray) -> None:
        # Face mode (mode 3): speak the recognized person's name directly.
        # If nobody is detected, stay silent. No mode announcement at all.
        if not msg.detections:
            return
        best = max(msg.detections, key=lambda d: float(d.confidence))
        name = best.class_name.strip()
        if not name:
            return
        now = time.monotonic()
        # New face -> announce right away; same face -> wait for the repeat gap
        # so we don't say the same name every frame.
        if name == self._last_face_name and now < self._face_announce_until:
            return
        self.get_logger().info(f"Face recognized: {name} ({best.confidence:.2f})")
        # The name is the same in both languages; announce it directly (this
        # also shows it on the web dashboard via /oak/announcement).
        self._say(name, name, key="face", priority=PRIO_RESULT)
        self._last_face_name = name
        self._face_announce_until = now + self._face_repeat_sec

    def _check_undervoltage(self) -> None:
        """Poll `vcgencmd get_throttled`; warn the user when the Pi is under-voltage.

        Output looks like `throttled=0x50005`. Bit 0 set = under-voltage is
        happening right now; bit 16 = under-voltage has occurred since boot.
        We alert on bit 0 (live condition) and re-alert every
        `undervoltage_repeat_sec` while it persists, so the user isn't spammed.
        """
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except Exception as exc:
            # A failure here (e.g. no permission on /dev/vcio) won't fix itself
            # mid-run, so warn once with a hint and stop polling to avoid spam.
            self.get_logger().warning(
                f"Under-voltage check failed; disabling monitor: {exc}. "
                "If this is a permission error on /dev/vcio, add the user to the "
                "'video' group: sudo usermod -aG video $USER (then reboot)."
            )
            if self._undervoltage_timer is not None:
                self._undervoltage_timer.cancel()
            return

        out = result.stdout.strip()
        if "=" not in out:
            return
        try:
            throttled = int(out.split("=", 1)[1], 0)
        except ValueError:
            return

        if not (throttled & 0x1):
            # No longer under-voltage; arm the next warning to fire immediately.
            self._undervoltage_warned_at = 0.0
            return

        now = time.monotonic()
        if now - self._undervoltage_warned_at < self._undervoltage_repeat_sec:
            return
        self._undervoltage_warned_at = now
        self.get_logger().warning(f"Raspberry Pi under-voltage detected ({out})")
        self._say(
            str(self.get_parameter("undervoltage_th").value),
            str(self.get_parameter("undervoltage_en").value),
            key="power", priority=PRIO_POWER,
        )

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
        self._say(speech_th, speech_en, key="mode", priority=PRIO_MODE)


def main(args=None) -> None:
    # Reach the user's PipeWire/PulseAudio session even when launched over SSH
    # (a remote shell lacks XDG_RUNTIME_DIR/PULSE_SERVER, which silences audio).
    applied = ensure_audio_session_env()
    rclpy.init(args=args)
    node = DecisionAudioNode()
    if applied:
        node.get_logger().info("Audio session env set for headless/SSH: " + ", ".join(applied))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        cv = getattr(node, "_speech_cv", None)
        if cv is not None:
            with cv:
                if node._current_proc is not None:
                    node._terminate_proc(node._current_proc)
                cv.notify_all()
        if hasattr(node, "_speech_thread"):
            node._speech_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
