from setuptools import find_packages
from setuptools import setup

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name="spikezoo",
    version="0.1.2",
    author="Kang Chen",
    author_email="mrchenkang@stu.pku.edu.cn",
    description="A deep learning toolbox for spike-to-image models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chenkang455/Spike-Zoo",
    packages=find_packages(),
    python_requires='>=3.7',
    include_package_data=True
)