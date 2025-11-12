"""
Setup script for Kelly Asset Pack Generator
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="kelly-pack",
    version="1.0.0",
    author="UI-TARS Team",
    description="8K digital human asset pipeline with open-source hair matting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kelly-pack",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "reportlab>=4.0.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "gpu": ["torch>=2.0.0", "torchvision>=0.15.0"],
        "video": ["imageio>=2.31.0", "imageio-ffmpeg>=0.4.9"],
        "dev": ["pytest>=7.4.0"],
        "all": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "imageio>=2.31.0",
            "imageio-ffmpeg>=0.4.9",
            "pytest>=7.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kelly-pack=kelly_pack.cli:main",
        ],
    },
)


