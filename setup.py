"""
KPU PyTorch Backend - C++ Extension Build Configuration

This setup.py builds the C++ extension for the KPU PyTorch backend.
The extension is built automatically when installing the package via pip install.
"""

import sys
from pathlib import Path

from setuptools import setup

# Minimum required PyTorch version
MIN_TORCH_VERSION = (2, 5, 0)

# Check for torch before proceeding
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError:
    print("PyTorch is required to build the KPU backend extension.")
    print(f"Install PyTorch first: pip install torch>={'.'.join(map(str, MIN_TORCH_VERSION))}")
    sys.exit(1)

# Check torch version
torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
if torch_version < MIN_TORCH_VERSION:
    print(f"PyTorch {'.'.join(map(str, MIN_TORCH_VERSION))} or later is required.")
    print(f"Found version: {torch.__version__}")
    sys.exit(1)

# Get the directory containing this setup.py (project root)
ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "kpu" / "torch" / "backend" / "csrc"
CSRC_DIR_REL = Path("kpu") / "torch" / "backend" / "csrc"

# Source files for the C++ extension (relative paths required by setuptools)
CPP_SOURCES = sorted(str(CSRC_DIR_REL / f.name) for f in CSRC_DIR.glob("*.cpp"))

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

# Extension name - always use full module path since building from root
EXT_NAME = "kpu.torch.backend._C"

# Define the extension module
ext_modules = [
    CppExtension(
        name=EXT_NAME,
        sources=CPP_SOURCES,
        include_dirs=[str(CSRC_DIR_REL)],
        extra_compile_args=CXX_FLAGS,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True),
    },
)
