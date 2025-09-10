# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from enum import Enum
from typing import Dict


class outDtype(Enum):
    INT8 = 0
    FLOAT = 1


LINEAR_PATTERNS: Dict = {
    "dq-linear-q": {
        "parent_nodes": ["aten::dequantize"],
        "child_nodes": ["aten::quantize_per_tensor"],
        "new_op": "pace::qlinear",
        "output_quantizer_idx": 0,
        "num_inputs_new_node": 6,
        "output_dtype": outDtype.INT8,
    },
    "dq-linear-relu-q": {
        "parent_nodes": ["aten::dequantize"],
        "child_nodes": ["aten::relu", "aten::quantize_per_tensor"],
        "new_op": "pace::qlinear_relu",
        "output_quantizer_idx": 1,
        "num_inputs_new_node": 6,
        "output_dtype": outDtype.INT8,
    },
    # The IPEX model does not fuse the output Quantizer
    # with the linear node, so we will not be doing it here
    "dq-linear-mul-add-q": {
        "parent_nodes": ["aten::dequantize"],
        "child_nodes": ["aten::mul", "aten::add"],
        "new_op": "pace::qlinear_mul_add",
        "output_quantizer_idx": 2,
        "num_inputs_new_node": 9,
        "output_dtype": outDtype.FLOAT,
    },
    "dq-linear-sigmoid": {
        "parent_nodes": ["aten::dequantize"],
        "child_nodes": ["aten::sigmoid"],
        "new_op": "pace::qlinear_sigmoid",
        "output_quantizer_idx": 0,
        "num_inputs_new_node": 6,
        "output_dtype": outDtype.FLOAT,
    },
}


def get_linear_patterns() -> Dict:
    return LINEAR_PATTERNS
