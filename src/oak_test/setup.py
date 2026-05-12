from setuptools import find_packages, setup


package_name = "oak_test"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    scripts=["scripts/oak_test_node"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/oak_test.yaml"]),
        (f"share/{package_name}/launch", ["launch/oak_test.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="abcd",
    maintainer_email="user@example.com",
    description="Simple ROS 2 Python test node that subscribes to camera frames and displays them.",
    license="Apache-2.0",
    tests_require=["pytest"],
)
