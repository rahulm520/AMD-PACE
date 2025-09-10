/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/ATen.h>
#include <graph/move_weights_to_memory.h>
#include <pace_tensor/pace_aten_interface.h>
#include <utils/graph_utils.h>

namespace pace {

// ,
//
/**
 * @brief Create a const node, and insert the const node into the graph,
 * and replace the input to the current node
 *
 * @param node Node to get the current subgraph
 * @param pace_tensor Tensor to be inserted as const node
 * @param debugname name to be given to the new node
 * @return torch::jit::Value*
 */
torch::jit::Value* insert_pace_tensor_as_const(
    torch::jit::Node* node,
    at::Tensor pace_tensor,
    std::string debugname) {
  auto graph = node->owningGraph();

  graph->setInsertPoint(node);

  torch::jit::Node* const_node =
      graph->create(string_to_symbol("prim::Constant"));
  const_node->output()->inferTypeFrom(pace_tensor);
  const_node->t_(c10::attr::value, std::move(pace_tensor));
  torch::jit::Value* pace_value_node = graph->insertNode(const_node)->output();
  pace_value_node->setDebugName(debugname);

  return pace_value_node;
}

/**
 * @brief Replaces the weights of the linear nodes with PACETensor
 *
 * @param node Node in which weights are to be replaced
 */
void replace_linear_weight_tensor_with_pace(torch::jit::Node* node) {
  torch::jit::Value* cpu_tensor = node->namedInput("weight");
  at::Tensor weight_cpu_tensor =
      torch::jit::constant_as<at::Tensor>(cpu_tensor).value();

  // matmul requires the weight to be of ba format
  at::Tensor weight_pace_tensor = create_pace_tensor_from_dense(
      weight_cpu_tensor, /*transpose*/ transpose_dims(0, 1));
  auto pace_value_node = insert_pace_tensor_as_const(
      node, weight_pace_tensor, cpu_tensor->debugName() + "_memory");
  node->replaceInputWith(cpu_tensor, pace_value_node);
}

/**
 * @brief Replaces the biases of the linear nodes with PACETensor
 *
 * @param node Node in which biases are to be replaced
 */
void replace_linear_bias_tensor_with_pace(torch::jit::Node* node) {
  torch::jit::Value* cpu_tensor = node->namedInput("bias");
  if (cpu_tensor->mustNotBeNone()) {
    at::Tensor bias_cpu_tensor =
        torch::jit::constant_as<at::Tensor>(cpu_tensor).value();

    // matmul requires the bias to be of the same dim as the output
    at::Tensor bias_pace_tensor = create_pace_tensor_from_dense(
        bias_cpu_tensor, /*transpose*/ transpose_dims(), /*ndims*/ 2);
    auto pace_value_node = insert_pace_tensor_as_const(
        node, bias_pace_tensor, cpu_tensor->debugName() + "_memory");
    node->replaceInputWith(cpu_tensor, pace_value_node);
  }
}

/**
 * @brief Wrapper for linear methods to insert the weight and bias PACETensors
 *
 * @param graph
 */
void move_linear_weights_to_pace(std::shared_ptr<torch::jit::Graph> graph) {
  std::vector<torch::jit::Node*> linear_nodes = findAllNodesFromGraph(
      graph->block(),
      {"pace::linear",
       "pace::linear_relu",
       "pace::qlinear",
       "pace::qlinear_relu",
       "pace::qlinear_mul_add",
       "pace::qlinear_sigmoid"});

  for (torch::jit::Node* node : linear_nodes) {
    replace_linear_weight_tensor_with_pace(node);
    replace_linear_bias_tensor_with_pace(node);
  }
}

void move_weights_to_memory(std::shared_ptr<torch::jit::Graph> graph) {
  // Move weights for all linear nodes
  move_linear_weights_to_pace(graph);
}

} // namespace pace
