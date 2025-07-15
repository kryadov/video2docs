#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="video2docs",
    version="0.1.0",
    author="Video2Docs Contributors",
    author_email="your.email@example.com",
    description="Convert YouTube or local videos to document formats (ODT, DOCX, PDF)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video2docs",
    package_dir={"": "."},
    packages=["src"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Office/Business",
        "Topic :: Text Processing :: Markup",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "video2docs=src.video2docs:main",
        ],
    },
)
