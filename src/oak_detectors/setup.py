from pathlib import Path

from setuptools import find_packages, setup


package_name = "oak_detectors"
package_root = Path(__file__).parent


def package_files(relative_dir: str):
    base_dir = package_root / relative_dir
    if not base_dir.exists():
        return []

    packaged = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file():
            continue
        install_dir = f"share/{package_name}/{path.parent.relative_to(package_root)}"
        packaged.append((install_dir, [str(path.relative_to(package_root))]))
    return packaged


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
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
