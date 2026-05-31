import audioop
import contextlib
import ctypes
import ctypes.util
import io
import json
import queue
import re
import shutil
import subprocess
import threading
import wave
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from oak_interfaces.msg import Mode

from .audio_env import ensure_audio_session_env


ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
)


def _py_error_handler(filename, line, function, err, fmt):
    return None


class SpeechLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("speech_logger_node")

        self.declare_parameter("language", "th-TH")
        self.declare_parameter("backend", "arecord")
        self.declare_parameter("mode_cmd_topic", "/oak/mode_cmd")
        # คำสั่งเสียง "เปิด/ปิดระบบนำทาง" ส่งเป็น std_msgs/Bool ไปยัง main_pipeline
        # โดยไม่เปลี่ยนโหมด (ยังอยู่ในโหมดเดิน สิ่งกีดขวางยังตรวจอยู่)
        self.declare_parameter("nav_cmd_topic", "/oak/nav_cmd")
        # ใช้รู้ว่าตอนนี้อยู่โหมดอะไร — คำสั่งเปิด/ปิดนำทางทำได้เฉพาะตอน WALK
        self.declare_parameter("current_mode_topic", "/oak/current_mode")
        self.declare_parameter("language_topic", "/oak/language")
        self.declare_parameter("default_language", "th")
        self.declare_parameter("walk_phrase", "เดินต่อไป")
        self.declare_parameter("money_phrase", "เช็คธนบัตร")
        self.declare_parameter("face_phrase", "สวัสดี")
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("tts_command", "edge-playback")
        self.declare_parameter("tts_voice_th", "th-TH-PremwadeeNeural")
        self.declare_parameter("tts_voice_en", "en-US-EmmaNeural")
        self.declare_parameter("announce_matched_commands", False)
        self.declare_parameter("device_index", -1)
        self.declare_parameter("arecord_device", "default")
        self.declare_parameter("sample_rate_hz", 16000)
        self.declare_parameter("log_microphones", True)
        self.declare_parameter("suppress_alsa_errors", True)
        self.declare_parameter("listen_timeout_sec", 5.0)
        self.declare_parameter("phrase_time_limit_sec", 2.0)
        self.declare_parameter("arecord_chunk_sec", 2.0)
        self.declare_parameter("ambient_duration_sec", 1.0)
        self.declare_parameter("energy_threshold", 300)
        self.declare_parameter("pause_threshold_sec", 0.8)
        self.declare_parameter("min_rms_threshold", 120)
        self.declare_parameter("log_silence_skips", False)

        self._recognition_language_th = "th-TH"
        self._recognition_language_en = "en-US"
        default_lang = str(self.get_parameter("default_language").value).strip().lower()
        self._app_language = "en" if default_lang.startswith("en") else "th"
        self._language = self._recognition_language_th if self._app_language == "th" else self._recognition_language_en
        self._backend = str(self.get_parameter("backend").value).strip().lower()
        self._mode_cmd_topic = str(self.get_parameter("mode_cmd_topic").value)
        self._nav_cmd_topic = str(self.get_parameter("nav_cmd_topic").value)
        self._current_mode_topic = str(self.get_parameter("current_mode_topic").value)
        self._language_topic = str(self.get_parameter("language_topic").value)
        # โหมดปัจจุบัน เริ่มที่ WALK (โหมดเริ่มต้นตอนเปิดเครื่อง) อัปเดตจาก /oak/current_mode
        self._current_mode = Mode.WALK
        self._walk_phrase = str(self.get_parameter("walk_phrase").value).strip().lower()
        self._money_phrases = {
            str(self.get_parameter("money_phrase").value).strip().lower(),
            "เช็กธนบัตร",
        }
        self._walk_phrases_en = {
            "walk mode",
            "keep walking",
            "walk",
            "go forward",
        }
        # Greeting a person triggers face-recognition mode (mode 3).
        self._face_phrases = {
            str(self.get_parameter("face_phrase").value).strip().lower(),
            "สวัสดี",
            "ดีครับ",
            "ดีค่ะ",
            "hello",
        }
        self._money_phrases.update(
            {
                "check banknote",
                "check the banknote",
                "money mode",
                "check money",
                "banknote mode",
            }
        )
        # คำสั่ง "ปิด" — ปิดระบบนำทาง (หยุดตรวจทางเดิน tactile) แต่ยังตรวจสิ่งกีดขวาง
        # แค่มีคำว่า "ปิด" อยู่ในประโยคก็พอ เช่น "ช่วยปิดระบบนำทางหน่อย"
        self._nav_off_phrases = {
            "ปิด",
            "navigation off",
            "turn off navigation",
            "disable navigation",
            "stop navigation",
        }
        # คำสั่ง "เปิด" — เปิดระบบนำทางกลับมา (เอาการตรวจทางเดิน tactile กลับมา)
        # แค่มีคำว่า "เปิด" อยู่ในประโยคก็พอ เช่น "เปิดระบบนำทางให้หน่อย"
        self._nav_on_phrases = {
            "เปิด",
            "navigation on",
            "turn on navigation",
            "enable navigation",
            "start navigation",
        }
        # English keyword tokens — matched as whole words, so the user only has to
        # say one keyword (e.g. just "money", "off") instead of the full phrase.
        # Token (not substring) matching is required because short words like "on"
        # are substrings of other words (e.g. "navigati-on").
        self._walk_keywords_en = {"walk", "forward"}
        self._money_keywords_en = {"money", "banknote", "cash"}
        self._face_keywords_en = {"hello", "hi", "hey"}
        self._nav_on_keywords_en = {"on", "enable", "start", "resume"}
        self._nav_off_keywords_en = {"off", "disable", "stop", "pause"}
        self._tts_enabled = bool(self.get_parameter("tts_enabled").value)
        self._tts_command = str(self.get_parameter("tts_command").value).strip()
        self._tts_voice_th = str(self.get_parameter("tts_voice_th").value).strip()
        self._tts_voice_en = str(self.get_parameter("tts_voice_en").value).strip()
        self._announce_matched_commands = bool(self.get_parameter("announce_matched_commands").value)
        self._device_index = int(self.get_parameter("device_index").value)
        self._arecord_device = str(self.get_parameter("arecord_device").value)
        self._sample_rate_hz = int(self.get_parameter("sample_rate_hz").value)
        self._log_microphones = bool(self.get_parameter("log_microphones").value)
        self._suppress_alsa_errors = bool(self.get_parameter("suppress_alsa_errors").value)
        self._listen_timeout_sec = self._optional_float("listen_timeout_sec")
        self._phrase_time_limit_sec = self._optional_float("phrase_time_limit_sec")
        self._arecord_chunk_sec = max(0.5, float(self.get_parameter("arecord_chunk_sec").value))
        self._ambient_duration_sec = max(0.0, float(self.get_parameter("ambient_duration_sec").value))
        self._energy_threshold = int(self.get_parameter("energy_threshold").value)
        self._pause_threshold_sec = max(0.0, float(self.get_parameter("pause_threshold_sec").value))
        self._min_rms_threshold = max(1, int(self.get_parameter("min_rms_threshold").value))
        self._log_silence_skips = bool(self.get_parameter("log_silence_skips").value)
        self._running = True
        self._alsa_error_handler = ERROR_HANDLER_FUNC(_py_error_handler)
        self._mode_publisher = self.create_publisher(Mode, self._mode_cmd_topic, 10)
        self._nav_publisher = self.create_publisher(Bool, self._nav_cmd_topic, 10)
        # Subscribe only: language is set at launch, but can still be overridden
        # at runtime via `ros2 topic pub /oak/language std_msgs/String "data: en"`.
        self.create_subscription(String, self._language_topic, self._on_language, 10)
        self.create_subscription(Mode, self._current_mode_topic, self._on_current_mode, 10)
        self._speech_queue = queue.Queue()
        self._tts_available = bool(self._tts_command) and shutil.which(self._tts_command) is not None

        if self._tts_enabled and not self._tts_available:
            self.get_logger().warning(
                f'TTS command "{self._tts_command}" is not installed. Speech logger voice feedback will be disabled.'
            )

        self._tts_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._tts_thread.start()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _optional_float(self, name: str) -> Optional[float]:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            return None
        return value

    @contextlib.contextmanager
    def _maybe_suppress_alsa_errors(self):
        if not self._suppress_alsa_errors:
            yield
            return

        lib_name = ctypes.util.find_library("asound")
        if not lib_name:
            yield
            return

        try:
            asound = ctypes.cdll.LoadLibrary(lib_name)
            asound.snd_lib_error_set_handler(self._alsa_error_handler)
        except Exception:
            yield
            return

        try:
            yield
        finally:
            try:
                asound.snd_lib_error_set_handler(None)
            except Exception:
                pass

    def _log_available_microphones(self, sr) -> None:
        if not self._log_microphones:
            return
        try:
            names = sr.Microphone.list_microphone_names()
        except Exception as exc:
            self.get_logger().warning(f"Failed to list microphones: {exc}")
            return
        if not names:
            self.get_logger().warning("No microphones detected by speech_recognition.")
            return
        for index, name in enumerate(names):
            self.get_logger().info(f"Microphone[{index}]: {name}")

    def _speech_loop(self) -> None:
        while self._running:
            try:
                utterance = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if utterance is None:
                return
            self._speak_item(utterance)

    def _announce_bilingual(self, text_th: str, text_en: str, log_message: Optional[str] = None) -> None:
        if log_message:
            self._safe_info(log_message)
        if self._app_language == "en":
            voice = self._tts_voice_en
            text = text_en
        else:
            voice = self._tts_voice_th
            text = text_th
        if not text:
            return
        try:
            while self._speech_queue.qsize() > 1:
                self._speech_queue.get_nowait()
        except queue.Empty:
            pass
        self._speech_queue.put_nowait([(text, voice)])

    def _on_language(self, msg: String) -> None:
        new_lang = (msg.data or "").strip().lower()
        if new_lang not in ("th", "en"):
            return
        if new_lang == self._app_language:
            return
        self._app_language = new_lang
        self._language = self._recognition_language_th if new_lang == "th" else self._recognition_language_en
        self._safe_info(f"Recognition language switched to {self._language}")

    def _on_current_mode(self, msg: Mode) -> None:
        self._current_mode = msg.mode

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
        if segments:
            try:
                while self._speech_queue.qsize() > 1:
                    self._speech_queue.get_nowait()
            except queue.Empty:
                pass
            self._speech_queue.put_nowait(segments)

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
            self._safe_warn(f"Speech logger voice feedback failed: {exc}")

    def _import_speech_recognition(self):
        try:
            return __import__("speech_recognition")
        except ImportError:
            self.get_logger().error(
                "speech_recognition is not installed. Activate workspace venv and pip install SpeechRecognition."
            )
            return None

    def _open_microphone(self, sr):
        selected_index = None if self._device_index < 0 else self._device_index
        with self._maybe_suppress_alsa_errors():
            return sr.Microphone(device_index=selected_index, sample_rate=self._sample_rate_hz)

    def _capture_with_pyaudio(self, sr):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self._energy_threshold
        recognizer.pause_threshold = self._pause_threshold_sec

        with self._maybe_suppress_alsa_errors():
            self._log_available_microphones(sr)

        try:
            microphone = self._open_microphone(sr)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open microphone with device_index={self._device_index}: {exc}"
            ) from exc

        self.get_logger().info(
            f"Speech logger is ready. Listening with pyaudio (device_index={self._device_index})."
        )

        while self._running and rclpy.ok():
            try:
                with self._maybe_suppress_alsa_errors():
                    with microphone as source:
                        try:
                            recognizer.adjust_for_ambient_noise(source, duration=self._ambient_duration_sec)
                        except Exception as exc:
                            self.get_logger().warning(f"Ambient noise calibration failed: {exc}")
                        audio = recognizer.listen(
                            source,
                            timeout=self._listen_timeout_sec,
                            phrase_time_limit=self._phrase_time_limit_sec,
                        )
            except sr.WaitTimeoutError:
                continue
            except Exception as exc:
                raise RuntimeError(f"Microphone listen failed: {exc}") from exc

            self._transcribe_audio(sr, recognizer, audio)

    def _capture_with_arecord(self, sr):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self._energy_threshold
        recognizer.pause_threshold = self._pause_threshold_sec

        self.get_logger().info(
            f"Speech logger is ready. Listening with arecord (device={self._arecord_device})."
        )

        while self._running and rclpy.ok():
            duration = self._arecord_chunk_sec
            duration_sec = max(1, int(round(duration)))
            cmd = [
                "arecord",
                "-q",
                "-D",
                self._arecord_device,
                "-f",
                "S16_LE",
                "-r",
                str(self._sample_rate_hz),
                "-c",
                "1",
                "-d",
                str(duration_sec),
                "-t",
                "wav",
                "-",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, check=True)
            except Exception as exc:
                if not self._running or not rclpy.ok():
                    return
                stderr = ""
                if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                    stderr = exc.stderr.decode("utf-8", errors="replace").strip()
                if stderr:
                    self._safe_warn(f"arecord capture failed on device={self._arecord_device}: {stderr}")
                else:
                    self._safe_warn(f"arecord capture failed on device={self._arecord_device}: {exc}")
                continue

            try:
                with sr.AudioFile(io.BytesIO(result.stdout)) as source:
                    audio = recognizer.record(source)
            except Exception as exc:
                self._safe_warn(f"Failed to decode arecord audio: {exc}")
                continue

            rms = self._calculate_wav_rms(result.stdout)
            if rms < self._min_rms_threshold:
                if self._log_silence_skips:
                    self._safe_info(f"Skipping low-energy audio chunk (rms={rms}).")
                continue

            self._transcribe_audio(sr, recognizer, audio)

    def _calculate_wav_rms(self, wav_bytes: bytes) -> int:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_width = wav_file.getsampwidth()
            if not frames or sample_width <= 0:
                return 0
            return int(audioop.rms(frames, sample_width))
        except Exception:
            return 0

    def _transcribe_audio(self, sr, recognizer, audio) -> None:
        try:
            transcript = recognizer.recognize_google(audio, language=self._language).strip()
        except sr.UnknownValueError:
            self._safe_info("Speech heard but could not be transcribed.")
            return
        except sr.RequestError as exc:
            self.get_logger().error(f"Speech recognition request failed: {exc}")
            return
        except (json.JSONDecodeError, ValueError):
            self._safe_info("Skipping malformed speech recognition response.")
            return
        except Exception as exc:
            self._safe_warn(f"Speech transcription failed: {exc}")
            return

        if transcript:
            self._safe_info(f'Speech: "{transcript}"')
            self._publish_mode_from_transcript(transcript)

    def _publish_mode_from_transcript(self, transcript: str) -> None:
        # Language is fixed at launch (launch arg `language:=en|th`); it is no
        # longer switchable by voice. Only mode commands are matched here.
        normalized = transcript.strip().lower()
        # English speech is matched on whole words ("tokens") so the user only has
        # to say one keyword instead of the full phrase. Thai phrases are still
        # matched as substrings (Thai words run together without spaces).
        tokens = set(re.findall(r"[a-z]+", normalized))
        # Navigation on/off is a sub-toggle of walk mode (tactile only); it does
        # not change mode, so match it before the mode phrases. Only the bare word
        # "เปิด"/"ปิด" (or one English keyword) is required, anywhere in the sentence.
        # Gated to WALK mode only — in money/face mode these commands are ignored.
        # "เปิด" (on) MUST be checked first because "ปิด" (off) is a substring of
        # "เปิด" — otherwise saying "เปิด" would wrongly trigger off.
        if self._current_mode == Mode.WALK:
            if any(phrase and phrase in normalized for phrase in self._nav_on_phrases) or (tokens & self._nav_on_keywords_en):
                self._publish_nav(True, transcript, "เปิดระบบนำทาง", "Navigation on")
                return
            if any(phrase and phrase in normalized for phrase in self._nav_off_phrases) or (tokens & self._nav_off_keywords_en):
                self._publish_nav(False, transcript, "ปิดระบบนำทาง", "Navigation off")
                return
        if any(phrase and phrase in normalized for phrase in {self._walk_phrase, *self._walk_phrases_en}) or (tokens & self._walk_keywords_en):
            self._publish_mode(Mode.WALK, transcript, "เดินต่อไป", "Walk mode")
            return
        if any(phrase and phrase in normalized for phrase in self._money_phrases) or (tokens & self._money_keywords_en):
            self._publish_mode(Mode.MONEY, transcript, "เช็คธนบัตร", "Money mode")
            return
        if any(phrase and phrase in normalized for phrase in self._face_phrases) or (tokens & self._face_keywords_en):
            # Face mode: switch silently, no mode announcement (the names spoken
            # by decision_audio are the only feedback the user gets).
            self._publish_mode(Mode.FACE, transcript, "ทักทาย", "Greeting", announce=False)
            return
        self._announce_bilingual("พูดใหม่", "Say again", f'Unrecognized speech command from "{transcript}".')

    def _publish_mode(self, mode_value: int, transcript: str, command_label_th: str, command_label_en: str, announce: bool = True) -> None:
        msg = Mode()
        msg.mode = mode_value
        self._mode_publisher.publish(msg)
        mode_name = {Mode.WALK: "WALK(0)", Mode.MONEY: "MONEY(1)", Mode.FACE: "FACE(2)"}.get(mode_value, str(mode_value))
        self._safe_info(f'Voice command matched "{command_label_th} / {command_label_en}" -> publish {mode_name} from "{transcript}"')
        if announce and self._announce_matched_commands:
            self._announce_bilingual(command_label_th, command_label_en)

    def _publish_nav(self, enabled: bool, transcript: str, command_label_th: str, command_label_en: str) -> None:
        msg = Bool()
        msg.data = enabled
        self._nav_publisher.publish(msg)
        self._safe_info(
            f'Voice command matched "{command_label_th} / {command_label_en}" '
            f'-> publish nav={enabled} from "{transcript}"'
        )
        # Always announce nav on/off so the user gets feedback for the toggle,
        # even when announce_matched_commands is off for mode switches.
        self._announce_bilingual(command_label_th, command_label_en)

    def _safe_info(self, message: str) -> None:
        if self._running and rclpy.ok():
            self.get_logger().info(message)

    def _safe_warn(self, message: str) -> None:
        if self._running and rclpy.ok():
            self.get_logger().warning(message)

    def _listen_loop(self) -> None:
        sr = self._import_speech_recognition()
        if sr is None:
            return

        if self._backend == "arecord":
            self._capture_with_arecord(sr)
            return

        if self._backend == "pyaudio":
            try:
                self._capture_with_pyaudio(sr)
            except Exception as exc:
                self.get_logger().error(str(exc))
            return

        try:
            self._capture_with_pyaudio(sr)
        except Exception as exc:
            self.get_logger().warning(f"{exc}. Falling back to arecord backend.")
            self._capture_with_arecord(sr)


def main(args=None) -> None:
    # Reach the user's PipeWire/PulseAudio session even when launched over SSH
    # (a remote shell lacks XDG_RUNTIME_DIR/PULSE_SERVER, which breaks the mic).
    applied = ensure_audio_session_env()
    rclpy.init(args=args)
    node = SpeechLoggerNode()
    if applied:
        node.get_logger().info("Audio session env set for headless/SSH: " + ", ".join(applied))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        if hasattr(node, "_speech_queue"):
            node._speech_queue.put_nowait(None)
        if hasattr(node, "_tts_thread"):
            node._tts_thread.join(timeout=1.0)
        if hasattr(node, "_thread"):
            node._thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
