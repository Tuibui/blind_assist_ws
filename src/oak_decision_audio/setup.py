from setuptools import find_packages, setup


package_name = "oak_decision_audio"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    scripts=["scripts/decision_audio_node", "scripts/speech_logger_node"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/oak_decision_audio.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="abcd",
    maintainer_email="user@example.com",
    description="Python decision and audio output skeleton for blind assist guidance.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
