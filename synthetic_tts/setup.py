#!/usr/bin/env python3
"""
Setup script for the Synthetic Digital TTS System.

This script installs the package and sets up the environment.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="synthetic-tts",
    version="1.0.0",
    author="Synthetic TTS Team",
    author_email="team@synthetic-tts.com",
    description="A completely self-contained, vendorless Text-to-Speech system for synthetic digital characters",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/synthetic-tts/synthetic-tts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "synthetic-tts=synthesize_speech:main",
            "synthetic-tts-train=train_models:main",
            "synthetic-tts-demo=scripts.demo:main",
            "synthetic-tts-setup=scripts.setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
    },
    zip_safe=False,
)








































