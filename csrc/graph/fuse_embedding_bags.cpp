/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/ATen.h>
#include <graph/fuse_embedding_bags.h>
#include <pace_tensor/pace_aten_interface.h>
#include <utils/graph_utils.h>

namespace pace {

#define CHECK_OK(value) \
  if (!value)           \
  return false

/**
 * @brief This method will check if the following pattern is found
 * and if found, aggregate all the quantized::embedding_bag_[byte/4bit]
 * nodes into a list and return it
 *
 * %1, %2, %3, .. = prim::ListUnpack(indices)
 * %4, %5, %6, .. = prim::ListUnpack(offsets)
 * %7 = any_op
 * %11 = quantized::embedding_bag_[byte/4bit](%w0, %1, %4, ...)
 * %12 = quantized::embedding_bag_[byte/4bit](%w1, %2, %5, ...)
 * %13 = quantized::embedding_bag_[byte/4bit](%w2, %3, %6, ...)
 * ...
 * %99 = prim::ListConstruct(%7, %11, %12, %13, ...)
 * aten::cat(%99, %5)
 *
 * @param concat_node A concat node
 * @return std::vector<torch::jit::Node*> list of
 * quantized::embedding_bag_[byte/4bit]
 */
std::vector<torch::jit::Node*> find_embeddigbag_cat_pattern(
    torch::jit::Node* concat_node,
    std::string kind) {
  std::vector<torch::jit::Node*> embedding_bag_nodes;
  torch::jit::Node* concat_input = concat_node->inputs().at(0)->node();
  if (concat_input->kind() == string_to_symbol("prim::ListConstruct")) {
    // It should have a minimum of two inputs, one Tensor from
    // any op and the others quantized::embedding_bag_[byte/4bit]
    // Iterate through the nodes and check if they are embedding bags
    at::ArrayRef<torch::jit::Value*> list_inputs = concat_input->inputs();
    if (list_inputs.size() > 2) {
      // The first input to the list should be a Tensor
      // if(list_inputs.at(0)->type() != c10::TensorType::get()) {
      // For some reason, direct comparison is not working, so checking the
      // strings
      if (list_inputs.at(0)->type()->repr_str() !=
          c10::TensorType::get()->repr_str()) {
        return {};
      }

      // Gather all the quantized::embedding_bag_[byte/4bit] nodes
      for (int idx = 1; idx < list_inputs.size(); idx++) {
        if (is_kind_at_idx(list_inputs, idx, kind)) {
          torch::jit::Node* embedding_bag_node = list_inputs.at(idx)->node();
          // Check if the index and offsets are from ListUnpack
          if (is_kind_at_idx(embedding_bag_node, 1, "prim::ListUnpack") &&
              is_kind_at_idx(embedding_bag_node, 2, "prim::ListUnpack")) {
            embedding_bag_nodes.emplace_back(embedding_bag_node);
          } // If ListUnpack
        } // If embedding_bag_[byte/4bit]
      } // Iterate through inputs
    } // If list_inputs.size() > 2
  } // If ListConstruct
  return embedding_bag_nodes;
}

/**
 * @brief Check if the following conditions are true, if any of them fails, do
 * not fuse For the conact node: axis = 1 For all the embedding bag nodes:
 * 1. scale_grad_by_freq = false
 * 2. mode = 0
 * 3. pruned_weights = false
 * 4. per_sample_weights_ = null
 * 5. compressed_indices_mapping = null
 * 6. include_last_offset = true
 *
 * @param embedding_bag_nodes list of the embedding bag nodes found
 * @return true
 * @return false
 */
bool check_constants(
    torch::jit::Node* conact_node,
    std::vector<torch::jit::Node*> embedding_bag_nodes) {
  bool fuse_ok = true;

  fuse_ok &= is_equal_at_idx<int64_t>(conact_node, 1, 1);
  CHECK_OK(fuse_ok);

  for (auto embedding_bag_node : embedding_bag_nodes) {
    fuse_ok &= is_equal_at_idx<bool>(embedding_bag_node, 3, false) &&
        is_equal_at_idx<int64_t>(embedding_bag_node, 4, 0) &&
        is_equal_at_idx<bool>(embedding_bag_node, 5, false) &&
        is_empty_at_idx<c10::optional<at::Tensor>>(embedding_bag_node, 6) &&
        is_empty_at_idx<c10::optional<at::Tensor>>(embedding_bag_node, 7) &&
        is_equal_at_idx<bool>(embedding_bag_node, 8, true);
    CHECK_OK(fuse_ok);
  }

  return fuse_ok;
}

#undef CHECK_OK

void fuse_embedding_bags_for_kind(
    std::shared_ptr<torch::jit::Graph> graph,
    std::string kind,
    int bit_width) {
  std::vector<torch::jit::Node*> concat_found_nodes = torch::jit::findAllNodes(
      graph->block(), string_to_symbol("aten::cat"), true);

  // Iterate through all concats
  for (torch::jit::Node* conact_node : concat_found_nodes) {
    std::vector<torch::jit::Node*> embedding_bag_nodes =
        find_embeddigbag_cat_pattern(conact_node, kind);
    if (embedding_bag_nodes.size()) {
      // Check the constants, if they fail, continue to the next concat
      // @TODO: Add one more check to check if the indices are 1D and offsets
      // have value
      if (!check_constants(conact_node, embedding_bag_nodes)) {
        continue;
      }

      graph->setInsertPoint(conact_node);

      // Gather all the packed_weights
      std::vector<torch::jit::Value*> embedding_bag_packed_weights;
      for (torch::jit::Node* embedding_bag_node : embedding_bag_nodes) {
        embedding_bag_packed_weights.emplace_back(
            embedding_bag_node->inputs().at(0));
      }

      torch::jit::Node* list_node = conact_node->inputs().at(0)->node();

      // Create new op with all the required inputs
      torch::jit::Node* qmerged_embbag_cat_packed = graph->create(
          string_to_symbol("pace::qmerged_embedding_bag_nbit_cat"));

      // Insert a list op which collects and lists all the packed weights
      auto packed_weights_type =
          embedding_bag_nodes[0]->inputs().at(0)->node()->output()->type();
      torch::jit::Node* packed_weights =
          graph->createList(packed_weights_type, embedding_bag_packed_weights);
      torch::jit::Value* packed_weights_output =
          graph->insertNode(packed_weights)->output();
      qmerged_embbag_cat_packed->addInput(
          packed_weights_output); // packed weights

      // Get the node for the indices and offsets
      torch::jit::Value* list_of_index =
          embedding_bag_nodes[0]->inputs().at(1)->node()->inputs().at(0);
      qmerged_embbag_cat_packed->addInput(list_of_index); // indices
      torch::jit::Value* list_of_offset =
          embedding_bag_nodes[0]->inputs().at(2)->node()->inputs().at(0);
      qmerged_embbag_cat_packed->addInput(list_of_offset); // offsets

      // Grab the first input of the list node
      torch::jit::Value* dense = list_node->inputs().at(0);
      qmerged_embbag_cat_packed->addInput(dense); // dense

      qmerged_embbag_cat_packed->addInput(
          graph->insertConstant(bit_width)); // bit_width

      // Insert the node into graph and replace the output of conact
      // with the output of the new node
      torch::jit::Value* qmerged_embbag_cat_packed_node =
          graph->insertNode(qmerged_embbag_cat_packed)->output();
      conact_node->output()->replaceAllUsesWith(qmerged_embbag_cat_packed_node);

      // Destroy the unwanted nodes
      conact_node->destroy();
      list_node->destroy();
      for (torch::jit::Node* embedding_bag_node : embedding_bag_nodes) {
        embedding_bag_node->destroy();
      }
    } // if embedding bags found
  } // for concat
}

void fuse_embedding_bags(std::shared_ptr<torch::jit::Graph> graph) {
  fuse_embedding_bags_for_kind(graph, "quantized::embedding_bag_byte", 8);
  fuse_embedding_bags_for_kind(graph, "quantized::embedding_bag_4bit", 4);
}

} // namespace pace
