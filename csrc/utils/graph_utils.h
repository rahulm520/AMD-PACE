/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef GRAPH_UTILS
#define GRAPH_UTILS

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <vector>

namespace pace {

/**
 * @brief Converts an op name from string to c10::Symbol
 *
 * @param op_name Name of the op in string format
 * @return c10::Symbol
 */
inline c10::Symbol string_to_symbol(std::string op_name) {
  return c10::Symbol::fromQualString(op_name);
}

inline void check_size(torch::jit::Node* node, int idx) {
  TORCH_CHECK(
      (node->inputs().size() > idx),
      "The node",
      node,
      " does not have idx: ",
      idx,
      ". Please confirm before accessing the input.");
}

/**
 * @brief Extract the constant value from the node
 *
 * @tparam T type of the value of the node
 * @param node node from which the constant need to be extraced
 * @param idx index of the input
 * @return T the value
 */
template <typename T>
inline T input_at_idx(torch::jit::Node* node, int idx) {
  check_size(node, idx);
  return torch::jit::constant_as<T>(node->inputs().at(idx)).value();
}

/**
 * @brief Check if the value at the index given is equal to the value provided
 *
 * @tparam T type of the value of the node
 * @param node node to check
 * @param idx index of the input
 * @param B To check the value against
 * @return true
 * @return false
 */
template <typename T>
inline bool is_equal_at_idx(torch::jit::Node* node, int idx, T B) {
  check_size(node, idx);
  return (input_at_idx<T>(node, idx) == B);
}

/**
 * @brief Check if the input at the given index of the node is of the specified
 * kind
 *
 * @param node node to check
 * @param idx index of the input
 * @param op_kind string value with the kind of the op
 * @return true
 * @return false
 */
inline bool is_kind_at_idx(
    torch::jit::Node* node,
    int idx,
    std::string op_kind) {
  check_size(node, idx);
  return node->inputs().at(idx)->node()->kind() == string_to_symbol(op_kind);
}

/**
 * @brief Check if the input at the given index of the node is of the specified
 * kind
 *
 * @param values inputs of the node
 * @param idx index of the input
 * @param op_kind string value with the kind of the op
 * @return true
 * @return false
 */
inline bool is_kind_at_idx(
    at::ArrayRef<torch::jit::Value*> values,
    int idx,
    std::string op_kind) {
  return values.at(idx)->node()->kind() == string_to_symbol(op_kind);
}

/**
 * @brief Check if the input at the index given is null
 *
 * @tparam T type of the value of the node
 * @param node node to check
 * @param idx index of the input
 * @return true
 * @return false
 */
template <typename T>
inline bool is_empty_at_idx(torch::jit::Node* node, int idx) {
  check_size(node, idx);
  return !(input_at_idx<T>(node, idx).has_value());
}

/**
 * @brief Find all instances of the nodes provided
 *
 * @param block Subgraph form where the ops should be searched
 * @param node_types Name of the node as string format
 * @return std::vector<torch::jit::Node*>
 */
inline std::vector<torch::jit::Node*> findAllNodesFromGraph(
    torch::jit::Block* block,
    std::vector<std::string> node_types) {
  std::vector<torch::jit::Node*> all_found_nodes;
  for (auto node_type : node_types) {
    std::vector<torch::jit::Node*> found_nodes =
        torch::jit::findAllNodes(block, string_to_symbol(node_type), true);
    all_found_nodes.insert(
        all_found_nodes.end(), found_nodes.begin(), found_nodes.end());
  }
  return all_found_nodes;
}

} // namespace pace

#endif // GRAPH_UTILS
