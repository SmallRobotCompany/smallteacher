from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = "\n".join(
    [line for line in long_description.split("\n") if not line.startswith("<img")]
)


setup(
    name="smallteacher",
    description=(
        "Research codebase for teacher-student based semi-supervised "
        "object detection in agricultural settings"
    ),
    author="Gabriel Tseng",
    author_email="gabrieltseng95@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smallrobotcompany/smallteacher",
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    packages=["smallteacher"]
    + [f"smallteacher.{f}" for f in find_packages("smallteacher")],
    install_requires=[
        # https://github.com/pytorch/pytorch/issues/78362
        "protobuf==3.20.1",
        "tqdm>=4.61.1",
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "pytorch-lightning==1.6.1",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
