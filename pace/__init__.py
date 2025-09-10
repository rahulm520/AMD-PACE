# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from .version import __version__

try:
    import torch  # noqa F401
except ModuleNotFoundError:
    raise ModuleNotFoundError("Torch not found, install torch. Refer to README.md.")
from . import _C as core
from . import utils
from . import llm
from . import ops

__all__ = ["__version__", "core", "utils", "llm", "ops"]
