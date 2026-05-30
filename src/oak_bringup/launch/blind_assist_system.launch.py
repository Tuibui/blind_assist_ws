"""ไฟล์ launch หลักของระบบช่วยเหลือผู้พิการทางสายตา

สั่งเปิด 4 node พร้อมกัน:
  1. main_pipeline      (oak_detectors)      — กล้อง OAK-D + ตรวจจับ (เดิน/เงิน/ใบหน้า)
  2. decision_audio     (oak_decision_audio) — ตัดสินใจว่าจะพูดอะไร แล้วออกเสียง
  3. speech_logger      (oak_decision_audio) — ฟังเสียงผู้ใช้ สั่งสลับโหมดด้วยคำพูด
  4. web_display        (oak_decision_audio) — หน้าเว็บแสดงภาพ/โหมด/ข้อความเตือน

วิธีสั่งเปิด:
  ros2 launch oak_bringup blind_assist_system.launch.py
  ros2 launch oak_bringup blind_assist_system.launch.py language:=en
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    # หา path ไฟล์ config (.yaml) ของแต่ละแพ็กเกจ จากโฟลเดอร์ share ที่ติดตั้งไว้
    detector_config = os.path.join(
        get_package_share_directory("oak_detectors"), "config", "main_pipeline.yaml"
    )
    decision_audio_config = os.path.join(
        get_package_share_directory("oak_decision_audio"), "config", "oak_decision_audio.yaml"
    )

    # ภาษาพูด: กำหนดครั้งเดียวตอน launch เช่น
    #   ros2 launch oak_bringup blind_assist_system.launch.py language:=en
    # รับค่า en/english หรือ th/thai ค่านี้จะไปทับ default_language ใน YAML
    # ของทั้ง 2 node เสียง (ไม่ต้องสั่งด้วยเสียง)
    language_arg = DeclareLaunchArgument(
        "language",
        default_value="th",
        description="Spoken/announcement language: en (english) or th (thai).",
    )
    language = LaunchConfiguration("language")

    # แหล่งภาพสำหรับตรวจ "ทางเดิน (tactile)" ในโหมดเดิน
    # ค่าว่าง "" = ใช้กล้อง OAK จริง (ค่าเริ่มต้น)
    # ถ้าใส่ path ไฟล์วิดีโอ จะอ่านภาพจากไฟล์นั้นแทน (ไม่ใช้กล้อง OAK และไม่มี
    # การเตือนสิ่งกีดขวางจาก depth) ใช้สำหรับทดสอบ/ดีบัก เช่น
    #   ros2 launch oak_bringup blind_assist_system.launch.py \
    #       tactile_video_source:=/home/abcd/Downloads/video.mp4
    tactile_video_source_arg = DeclareLaunchArgument(
        "tactile_video_source",
        default_value="",
        description="Walk-mode tactile input: empty = OAK camera (default), or a video file path.",
    )
    tactile_video_source = LaunchConfiguration("tactile_video_source")

    return LaunchDescription(
        [
            language_arg,
            tactile_video_source_arg,
            # node กล้อง+ตรวจจับหลัก: ส่ง config + แหล่งวิดีโอ tactile (ถ้ามี) ให้
            Node(
                package="oak_detectors",
                executable="main_pipeline",
                name="main_pipeline_node",
                output="screen",
                parameters=[detector_config, {"tactile_video_source": tactile_video_source}],
            ),
            # node ตัดสินใจ+ออกเสียง: รับภาษาที่เลือกตอน launch ไปทับค่าใน config
            Node(
                package="oak_decision_audio",
                executable="decision_audio_node",
                name="decision_audio_node",
                output="screen",
                parameters=[decision_audio_config, {"default_language": language}],
            ),
            # node ฟังเสียงสั่งงาน: ภาษาที่รู้จำ (ไทย/อังกฤษ) ตามภาษาที่เลือก
            Node(
                package="oak_decision_audio",
                executable="speech_logger_node",
                name="speech_logger_node",
                output="screen",
                parameters=[decision_audio_config, {"default_language": language}],
            ),
            # node หน้าเว็บแสดงผล (ไม่มีหน้าต่าง GUI) เปิดที่ http://<ip>:8080
            Node(
                package="oak_decision_audio",
                executable="web_display_node",
                name="web_display_node",
                output="screen",
                parameters=[decision_audio_config],
            ),
        ]
    )
