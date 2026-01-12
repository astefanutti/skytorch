"""
KPU PyTorch Backend - Build Configuration

This setup.py builds the C++ extension for the KPU PyTorch backend.
The extension provides the low-level device management required for the backend.

Usage:
    # Build the extension in-place (from this directory)
    cd kpu/torch/backend
    python setup.py build_ext --inplace --build-lib .

    # Or build from project root (copies to kpu/torch/backend/)
    python kpu/torch/backend/setup.py build_ext --inplace --build-lib .
"""

import sys
from pathlib import Path

from setuptools import setup

# Check for torch before proceeding
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError:
    print("PyTorch is required to build the KPU backend extension.")
    print("Install PyTorch first: pip install torch>=2.1.0")
    sys.exit(1)

# Get the directory containing this setup.py
ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "csrc"

# Source files for the C++ extension
CPP_SOURCES = sorted(CSRC_DIR.glob("*.cpp"))

# Compiler flags
if sys.platform == "win32":
    CXX_FLAGS = ["/sdl", "/permissive-"]
else:
    CXX_FLAGS = [
        "-std=c++17",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-Wno-missing-field-initializers",
    ]

# Platform-specific flags
if sys.platform == "darwin":
    CXX_FLAGS.extend([
        "-mmacosx-version-min=10.14",
        "-Wno-unused-command-line-argument",
    ])
elif sys.platform == "linux":
    CXX_FLAGS.extend([
        "-fPIC",
        "-Wno-unused-but-set-variable",
    ])

# Determine extension name based on build location
# When running from the backend directory with --inplace, use short name
# so the .so file is placed directly in the current directory
import os
if os.getcwd() == str(ROOT_DIR):
    EXT_NAME = "_C"
else:
    EXT_NAME = "kpu.torch.backend._C"

# Define the extension module
# Use absolute paths to support building from any directory
ext_modules = [
    CppExtension(
        name=EXT_NAME,
        sources=[str(s) for s in CPP_SOURCES],
        include_dirs=[str(CSRC_DIR)],
        extra_compile_args=CXX_FLAGS,
    ),
]

setup(
    name="kpu-torch",
    version="0.1.0",
    description="KPU PyTorch Backend - Remote execution backend for Kubernetes",
    author="Antonin Stefanutti",
    author_email="antonin.stefanutti@gmail.com",
    url="https://github.com/astefanutti/kpu",
    ext_modules=ext_modules,
    cmdclass={
        # Use parallel compilation (uses all available cores by default)
        # Can be overridden with MAX_JOBS environment variable
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True),
    },
    entry_points={
        "torch.backends": [
            "kpu = kpu.torch.backend:_autoload",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={
        "kpu.torch.backend": ["*.so", "*.dylib", "*.dll"],
    },
)
