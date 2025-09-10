# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import Dict

import torch

from .quantized_patterns import outDtype
from .graph_utils import (
    parentAt_,
    children_,
    child_assume_one_,
    find_which_input,
    find_all_nodes_by_type_,
)


def merge_dq_linear_q(graph: torch.Graph, patterns: Dict):
    """
    This function will merge the dequantize, linear, (Optional) post-ops, and quantize
    nodes into a single node. This will be done for all the linear nodes in the graph.
    Supported patterns:
    1. dequantize -> linear -> quantize_per_tensor: int8 -> int8
    2. dequantize -> linear -> relu -> quantize_per_tensor: int8 -> int8
    3. dequantize -> linear -> mul -> add -> quantize_per_tensor: int8 -> int8
    4. dequantize -> linear -> sigmoid: int8 -> float32
    """

    # Find all linear nodes in the graph
    all_linear_list = find_all_nodes_by_type_(graph, "aten::linear")

    # Iterate through all the linear nodes
    for curr_node in all_linear_list:
        for key, pattern_map in patterns.items():
            pattern_found = True

            # Assumption: The linear node will only have one child node
            # Need to find the whole pattern, so iterate through all the
            # next nodes and check if they match the pattern
            next_node = child_assume_one_(curr_node)
            for i in range(len(pattern_map["child_nodes"])):
                if (
                    next_node is not None
                    and next_node.kind() != pattern_map["child_nodes"][i]
                ):
                    pattern_found = False
                next_node = child_assume_one_(next_node)

            if pattern_found:
                graph.setInsertPoint(curr_node)

                # Assumption: The linear node will only have one inputs node
                # and that the input node will be a dequant node,
                # If not, then there is some issue, assert it
                prev_node = parentAt_(curr_node, 0)  # This will be the dq node
                assert prev_node.kind() == pattern_map["parent_nodes"][0], (
                    "Previous node not dequantize," " not sure if it should be folded."
                )

                # Create a new node and add all the inputs
                new_node = graph.create(pattern_map["new_op"])
                new_node.addInput(
                    prev_node.inputsAt(0)
                )  # Get input from the dequant node
                new_node.addInput(
                    curr_node.inputsAt(1).node().inputsAt(0)
                )  # Get weight from dequant input of the current node
                new_node.addInput(
                    curr_node.inputsAt(2)
                )  # Get bias from the current node

                # Assumption: If the output should be INT8, the linear module
                # should have a quantize_per_tensor following it
                if pattern_map["output_dtype"] == outDtype.INT8:
                    output_quantizer_node = child_assume_one_(curr_node)
                    for i in range(pattern_map["output_quantizer_idx"]):
                        output_quantizer_node = child_assume_one_(output_quantizer_node)

                    # assert that the quantizer node should be a quantize_per_tensor node
                    assert (
                        output_quantizer_node.kind() == "aten::quantize_per_tensor"
                    ), (
                        "If the output is INT8,"
                        "it should have quantize_per_tensor following it."
                    )
                    new_node.addInput(
                        output_quantizer_node.inputsAt(1)
                    )  # Get scale from the quant node
                    new_node.addInput(
                        output_quantizer_node.inputsAt(2)
                    )  # Get zero point from the quant node
                    new_node.addInput(
                        output_quantizer_node.inputsAt(3)
                    )  # Get dtype from the quant node??

                if key == "dq-linear-mul-add-q":
                    mul_node = child_assume_one_(curr_node)
                    add_node = child_assume_one_(mul_node)

                    new_node.addInput(
                        mul_node.inputsAt(0)
                    )  # Add the first node of mul to new node (binary_mul multiplier)
                    new_node.addInput(
                        add_node.inputsAt(1)
                    )  # Add the second node of mul to new node (binary_add addend)
                    new_node.addInput(
                        add_node.inputsAt(2)
                    )  # Add the second node of mul to new node (binary_add alpha)

                # Find the next logical node after quant and set it's input as new node
                # If the output is INT8, then the output node will be the node after quantize
                if pattern_map["output_dtype"] == outDtype.INT8:
                    output_node = child_assume_one_(output_quantizer_node)
                # If the output is FLOAT, then the output node will be the node after linear
                else:
                    output_node = child_assume_one_(curr_node)
                    for i in range(len(pattern_map["child_nodes"])):
                        output_node = child_assume_one_(output_node)

                new_node_output = new_node.output()
                output_node.replaceInput(0, new_node_output)

                # Insert new node into graph
                new_node.insertBefore(curr_node)

                # Clean-up part
                # Destroy nodes which are not required now
                # Add them in order starting from the last node
                nodes_to_destroy = []

                # SPECIAL CASE
                # Destroy quantizer right now so that any nodes
                # using the new node can be found and mapped easily
                # Eg: To support the special case of add node being reused in DCN
                if pattern_map["output_dtype"] == outDtype.INT8:
                    # nodes_to_destroy.append(output_quantizer_node)

                    # Check if the quant node is being reused anywhere
                    # If yes, then update that node to use the new node output
                    nodes_using_quant_node = children_(output_quantizer_node)
                    for node_using_quant_node in nodes_using_quant_node:
                        if node_using_quant_node is not None:
                            idx = find_which_input(
                                node_using_quant_node, output_quantizer_node
                            )
                            node_using_quant_node.replaceInput(idx, new_node_output)
                    output_quantizer_node.destroy()
                # Destroy add and mul nodes
                if key == "dq-linear-mul-add-q":
                    # SPECIAL CASE
                    # In case the add node is being reused anywhere,
                    # update that node to use the new node output
                    node_using_add_node = child_assume_one_(add_node)
                    if node_using_add_node is not None:
                        idx = find_which_input(node_using_add_node, add_node)
                        node_using_add_node.replaceInput(idx, new_node_output)

                    nodes_to_destroy.append(add_node)
                    nodes_to_destroy.append(mul_node)
                # For relu and sigmoid, delete the next node (aten::relu/aten::sigmoid)
                if key == "dq-linear-relu-q" or key == "dq-linear-sigmoid":
                    nodes_to_destroy.append(child_assume_one_(curr_node))

                nodes_to_destroy.append(curr_node)  # Destroy the current node
                nodes_to_destroy.append(
                    curr_node.inputsAt(1).node()
                )  # Destroy the weight dequant node
                nodes_to_destroy.append(prev_node)  # Destroy the dequant node

                # SPECIAL CASE
                # In case the dequant node is being reused anywhere,
                # update that node to use the new node output
                nodes_using_dequant_node = children_(prev_node)
                for node_using_dequant_node in nodes_using_dequant_node:
                    if node_using_dequant_node is not None:
                        idx = find_which_input(node_using_dequant_node, prev_node)
                        node_using_dequant_node.replaceInput(idx, prev_node.inputsAt(0))

                for node in nodes_to_destroy:
                    node.destroy()

                # break the loop so that the other patterns are not serached
                break

    # Graph cleanup
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
