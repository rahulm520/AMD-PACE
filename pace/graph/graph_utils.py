# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

from typing import List, Union

import torch


def find_all_nodes_by_type_(graph: torch.Graph, node_type: str) -> List[torch.Node]:
    """
    Returns all the nodes of the type passed in the graph from bottom to top
    """
    node_list = graph.findAllNodes(node_type)
    node_list.reverse()
    return node_list


def parentAt_(node: torch.Node, input_idx: int = 0) -> torch.Node:
    """
    Returns the node that is the parent of the node passed at the input index
    """
    return node.inputsAt(input_idx).node()


def children_(node: torch.Node) -> List[torch.Node]:
    """
    Returns all the nodes that are the children of the node passed
    """

    all_found_child_nodes = []

    # Iterate through all the nodes in the graph block
    for searching_node in node.owningBlock().nodes():
        # Iterate through all the inputs of these nodes
        for input_of_searching_node in searching_node.inputs():
            # Check if any of these inputs are the output from the node required
            if input_of_searching_node == node.output():
                all_found_child_nodes.append(searching_node)
    return all_found_child_nodes


def child_assume_one_(node: torch.Node) -> torch.Node:
    """
    Returns the child node assuming there is only one child
    """
    try:
        return children_(node)[0]
    except IndexError:
        return None
    except AttributeError:
        return None


def find_which_input(node: torch.Node, check_node: torch.Node) -> Union[int, None]:
    """
    Returns the index of the input of the node passed that is the output of the check node
    """

    for idx, input_of_node in enumerate(node.inputs()):
        if input_of_node == check_node.output():
            return idx
        idx += 1
    return None


def append_if_not_exist_(list_obj: List, value: torch.Node) -> List:
    """
    Appends the value to the list if it does not exist
    """
    if value not in list_obj:
        list_obj.append(value)
    return list_obj
