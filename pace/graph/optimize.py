# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import copy

import torch

from .quantized_patterns import get_linear_patterns
from .remove_alias import remove_alias_from_graph
from .qlinear import merge_dq_linear_q


def optimize_qdq(
    scripted_model: torch.jit._script.RecursiveScriptModule,
    inplace: bool = False,
) -> torch.jit._script.RecursiveScriptModule:
    """
    Optimize QDQ models into PACE models
    Args:
        scripted_model: The QDQ model to be converted.
        inplace: Whether to convert the model inplace.
    """
    # Make a copy of the scripted model if inplace is False
    if not inplace:
        zen_scripted_model = copy.deepcopy(scripted_model)
    else:
        zen_scripted_model = scripted_model

    # Get the graph from the scripted model
    graph = zen_scripted_model.graph

    # Remove the unwanted aliases from the graph
    remove_alias_from_graph(graph)

    # Merge DQ -> Linear -> * -> Q to compatible nodes
    merge_dq_linear_q(graph, get_linear_patterns())

    # Eliminate any dead code/node and lint the current graph
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    return zen_scripted_model
