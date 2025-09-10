# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import torch


def remove_alias_from_graph(graph: torch.Graph):
    """
    Removes all the alias nodes from the graph
    """

    alias_node_count = 0
    for node in graph.findAllNodes("aten::alias"):
        # Get the inputs and outputs of the node
        inputs = node.input()
        outputs = node.output()

        # Replace all the uses of the outputs with the inputs
        outputs.replaceAllUsesWith(inputs)

        # Remove the node from the graph
        node.destroy()

        alias_node_count += 1
