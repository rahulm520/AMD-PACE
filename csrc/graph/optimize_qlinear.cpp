/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <graph/optimize_qlinear.h>
#include <utils/graph_utils.h>

namespace pace {
std::vector<std::string> get_qlinear_mapping() {
  return {"quantized::linear", "quantized::linear_relu"};
}

const std::unordered_map<std::string, std::string>
    QLINEAR_TO_ZENQLINEAR_MAPPING = {
        {"quantized::linear", "pace::qlinear"},
        {"quantized::linear_relu", "pace::qlinear_relu"}};

const char* get_zenqlinear_node(const char* orig_op) {
  auto it = QLINEAR_TO_ZENQLINEAR_MAPPING.find(orig_op);
  if (it != QLINEAR_TO_ZENQLINEAR_MAPPING.end()) {
    return it->second.c_str();
  }
  return nullptr;
}
/**
 * @brief This method checks for the following node patterns and converts
 * them to their linear counter parts as provided by pace library
 * quantized::linear nodes are converted to pace::qlinear
 * quantized::linear_relu nodes are converted to pace::qlinear_relu nodes
 * @param graph model graph
 * @param node linear node which needs to be replaced
 * @param new_op new operation node name to which conversion needs to happen
 * @return None
 */
void replace_with_pace_qlinear(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Node* node,
    const char* new_op) {
  if (node and node->inputs().size() < 4) {
    return;
  }
  graph->setInsertPoint(node);
  auto packed_w_b = node->inputs().at(1)->node()->output();
  auto packed_w_b_converted =
      torch::jit::constant_as<c10::intrusive_ptr<LinearPackedParamsBase>>(
          packed_w_b)
          .value()
          ->unpack();
  at::Tensor weight = std::get<0>(packed_w_b_converted);
  // only 2D weights tensor supported for now
  if (weight.dim() != 2) {
    return;
  }
  c10::optional<at::Tensor> bias = std::get<1>(packed_w_b_converted);
  auto w_node = graph->insertConstant(weight);
  torch::jit::Value* b_node = nullptr;
  if (bias.has_value()) {
    at::Tensor biasValue = bias.value();
    torch::jit::Node* const_node =
        graph->create(string_to_symbol("prim::Constant"));
    const_node->output()->inferTypeFrom(biasValue);
    const_node->t_(c10::attr::value, biasValue);
    b_node = graph->insertNode(const_node)->output();
  } else {
    b_node = graph->insertConstant(bias);
  }

  auto new_node = graph->create(string_to_symbol(new_op));
  new_node->addInput(node->input(0));
  new_node->addInput(w_node);
  new_node->addInput(b_node);
  new_node->addInput(node->input(2));
  new_node->addInput(node->input(3));

  auto c = graph->insertConstant(13);
  new_node->addInput(c);

  auto new_inserted_node = graph->insertNode(new_node)->output();
  node->output()->replaceAllUsesWith(new_inserted_node);

  node->destroy();
}

/**
 * @brief This method checks for the following node patterns and converts
 * them to its linear counter part as provided by pace library
 * %1 : pace::qlinear
 * %2 : aten::dequantize
 * %3 : aten::sigmoid
 * Above pattern is fused into a single node pace::qlinear_sigmoid
 * @param graph model graph
 * @return None
 */
void fuse_qlinear_sigmoid(std::shared_ptr<torch::jit::Graph> graph) {
  const torch::jit::Symbol quantized_linear_symbol =
      string_to_symbol("pace::qlinear");
  const torch::jit::Symbol dequantize_symbol =
      string_to_symbol("aten::dequantize");
  const torch::jit::Symbol sigmoid_symbol = string_to_symbol("aten::sigmoid");

  auto linear_nodes =
      torch::jit::findAllNodes(graph->block(), quantized_linear_symbol, true);

  for (auto& linear_node : linear_nodes) {
    if (linear_node->outputs().size() == 1 &&
        linear_node->outputs().at(0)->uses().size() == 1 &&
        linear_node->outputs().at(0)->uses().at(0).user->kind() ==
            dequantize_symbol) {
      if (linear_node->inputs().size() < 3) {
        continue;
      }
      graph->setInsertPoint(linear_node);

      auto input = linear_node->input(0);
      auto weight = linear_node->input(1);
      auto bias = linear_node->input(2);

      auto qlinear_sigmoid_node =
          graph->create(string_to_symbol("pace::qlinear_sigmoid"));
      qlinear_sigmoid_node->addInput(input);
      qlinear_sigmoid_node->addInput(weight);
      qlinear_sigmoid_node->addInput(bias);

      auto dequantize_node = linear_node->outputs().at(0)->uses().at(0).user;
      auto sigmoid_node = dequantize_node->outputs().at(0)->uses().at(0).user;
      auto inserted_qlinear_sigmoid_node =
          graph->insertNode(qlinear_sigmoid_node)->output();
      sigmoid_node->output()->replaceAllUsesWith(inserted_qlinear_sigmoid_node);

      sigmoid_node->destroy();
      dequantize_node->destroy();
      linear_node->destroy();
    }
  }
}
void optimize_qlinear(std::shared_ptr<torch::jit::Graph> graph) {
  for (const auto& orig_op : pace::get_qlinear_mapping()) {
    const char* new_op = pace::get_zenqlinear_node(orig_op.c_str());
    auto all_linear_nodes = torch::jit::findAllNodes(
        graph->block(), string_to_symbol(orig_op), true);
    for (auto node : all_linear_nodes) {
      pace::replace_with_pace_qlinear(graph, node, new_op);
    }
  }
  fuse_qlinear_sigmoid(graph);
}
} // namespace pace
