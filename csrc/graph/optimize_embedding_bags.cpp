/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/ATen.h>
#include <graph/optimize_embedding_bags.h>

namespace pace {
/**
 * @brief This method will check if the following pattern is found
 * and if found, aggregate all the aten::quantize_per_tensor
 * nodes into a single aten::quantize_per_tensor node
 * Snippet of original graph :
 * %1 = aten::quantize_per_tensor(%emb_output_1, %scale, %zpoint, %dtype)
 * %2 = aten::quantize_per_tensor(%emb_output_2, %scale, %zpoint, %dtype)
 * %3 = aten::quantize_per_tensor(%emb_output_3, %scale, %zpoint, %dtype)
 * ...
 * %452 : prim::ListConstruct(%dense_arch_model__mlp_4.1, %1, %2, %3.....)
 * %cat.1 : aten::cat(%452, %454)
 * %459 : int[] = prim::ListConstruct(%86, %458)
 * %460 = aten::reshape(%cat.1, %459)
 * Snippet of converted graph :
 * %452 : prim::ListConstruct(%616, %emb_output_1, %emb_output_2, ........)
 * %cat.1 : aten::cat(%452, %454)
 * %459 : prim::ListConstruct(%86, %458)
 * %460 : aten::reshape(%cat.1, %459)
 * %615 : aten::quantize_per_tensor(%460, %97, %98, %61)
 * @param graph model graph
 * @return None
 */
void optimize_emb_quantize_per_tensor(
    std::shared_ptr<torch::jit::Graph> graph) {
  auto emb_nodes = torch::jit::findAllNodes(
      graph->block(), string_to_symbol("quantized::embedding_bag_byte"), true);
  if (emb_nodes.size() == 0) {
    return;
  }
  bool found_pattern = true;
  std::vector<torch::jit::Node*> quantize_per_tensor_nodes;
  for (const auto& emb_node : emb_nodes) {
    torch::jit::Node* output_user_node =
        emb_node->outputs().at(0)->uses().at(0).user;
    if (output_user_node->kind() ==
        string_to_symbol("aten::quantize_per_tensor")) {
      quantize_per_tensor_nodes.emplace_back(output_user_node);
    } else {
      found_pattern = false;
      break;
    }
  }
  if (!found_pattern) {
    return;
  }
  // we check here if all quantize_per_tensor nodes have same scale, zpoint and
  // dtype values
  auto q_per_tensor_scale = quantize_per_tensor_nodes[0]
                                ->input(1)
                                ->node()
                                ->t(c10::attr::value)
                                .item<float>();
  auto q_per_tensor_zpoint = quantize_per_tensor_nodes[0]
                                 ->input(2)
                                 ->node()
                                 ->t(c10::attr::value)
                                 .item<int64_t>();
  auto q_per_tensor_output_dtype =
      quantize_per_tensor_nodes[0]->input(3)->node()->i(c10::attr::value);

  for (int index = 1; index < emb_nodes.size(); index++) {
    auto curr_q_per_tensor_scale = quantize_per_tensor_nodes[index]
                                       ->input(1)
                                       ->node()
                                       ->t(c10::attr::value)
                                       .item<float>();
    auto curr_q_per_tensor_zpoint = quantize_per_tensor_nodes[index]
                                        ->input(2)
                                        ->node()
                                        ->t(c10::attr::value)
                                        .item<int64_t>();
    auto curr_q_per_tensor_output_dtype =
        quantize_per_tensor_nodes[index]->input(3)->node()->i(c10::attr::value);
    if (curr_q_per_tensor_scale != q_per_tensor_scale ||
        curr_q_per_tensor_zpoint != q_per_tensor_zpoint ||
        curr_q_per_tensor_output_dtype != q_per_tensor_output_dtype) {
      return;
    }
  }
  torch::jit::Node* list_construct_node =
      quantize_per_tensor_nodes.back()->outputs().at(0)->uses().at(0).user;
  if (list_construct_node->kind() != string_to_symbol("prim::ListConstruct")) {
    return;
  }
  torch::jit::Node* qlinear_relu_node =
      list_construct_node->inputs().at(0)->node();
  if (qlinear_relu_node->kind() != string_to_symbol("pace::qlinear_relu")) {
    return;
  }
  auto cat_node = list_construct_node->outputs().at(0)->uses().at(0).user;
  if (cat_node->kind() != string_to_symbol("aten::cat")) {
    return;
  }
  auto reshape_node = cat_node->outputs().at(0)->uses().at(0).user;
  if (reshape_node->kind() != string_to_symbol("aten::reshape")) {
    return;
  }
  for (int i = 0; i < emb_nodes.size(); ++i) {
    list_construct_node->replaceInput(i + 1, emb_nodes[i]->output());
  }
  auto reshape_outputs = reshape_node->outputs();
  auto new_quantize_node =
      graph->create(string_to_symbol("aten::quantize_per_tensor"));
  graph->setInsertPoint(reshape_node);
  new_quantize_node->insertAfter(reshape_node);
  reshape_node->output()->replaceAllUsesWith(new_quantize_node->output());

  new_quantize_node->addInput(reshape_node->output());
  new_quantize_node->addInput(quantize_per_tensor_nodes.back()->inputs().at(1));
  new_quantize_node->addInput(quantize_per_tensor_nodes.back()->inputs().at(2));
  new_quantize_node->addInput(quantize_per_tensor_nodes.back()->inputs().at(3));

  auto dequant_node = graph->create(string_to_symbol("aten::dequantize"));
  auto qlinear_relu_node_output = qlinear_relu_node->output();
  qlinear_relu_node->output()->replaceAllUsesWith(dequant_node->output());

  dequant_node->insertAfter(qlinear_relu_node);
  dequant_node->addInput(qlinear_relu_node_output);

  for (auto& quantize_per_tensor_node : quantize_per_tensor_nodes) {
    quantize_per_tensor_node->destroy();
  }
}
} // namespace pace
