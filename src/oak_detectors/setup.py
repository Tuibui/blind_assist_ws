# ไฟล์ติดตั้ง (setup) ของแพ็กเกจ oak_detectors — บอก colcon/ament ว่าจะติดตั้ง
# โค้ด, สคริปต์ตัวรัน node และไฟล์ข้อมูล (config/models) ไปไว้ที่ไหน
from pathlib import Path

from setuptools import find_packages, setup


package_name = "oak_detectors"
package_root = Path(__file__).parent


def package_files(relative_dir: str):
    """รวบรวมไฟล์ทั้งหมดใต้โฟลเดอร์ relative_dir (เช่น config, models) เพื่อให้
    ติดตั้งไปไว้ใน share/<package> โดยคงโครงสร้างโฟลเดอร์เดิม
    คืน list ของคู่ (โฟลเดอร์ปลายทาง, [ไฟล์ต้นทาง]) ตามรูปแบบที่ data_files ต้องการ"""
    base_dir = package_root / relative_dir
    if not base_dir.exists():
        return []

    packaged = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file():
            continue  # ข้ามโฟลเดอร์ เก็บเฉพาะไฟล์
        install_dir = f"share/{package_name}/{path.parent.relative_to(package_root)}"
        packaged.append((install_dir, [str(path.relative_to(package_root))]))
    return packaged


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    # ตัวรัน node เป็น bash wrapper (ดูเหตุผลใน HANDOFF.md — ไม่ใช้ console_scripts)
    scripts=["scripts/main_pipeline"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ]
    + package_files("config")
    + package_files("models"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="abcd",
    maintainer_email="user@example.com",
    description="Python detector node skeletons for money and navigation perception.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
