/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <graph/fuse_embedding_bags.h>
#include <graph/move_weights_to_memory.h>
#include <graph/optimize_embedding_bags.h>
#include <graph/optimize_for_inference.h>
#include <graph/optimize_qlinear.h>

namespace pace {

void Optimize(std::shared_ptr<torch::jit::Graph>& graph) {
  GRAPH_DUMP("Original Graph", graph);
  if (torch::jit::getProfilingMode()) {
    GRAPH_DUMP("Before RemoveProfileNodesAndSpecializeTypes", graph);
    RemoveProfileNodesAndSpecializeTypes(graph);
  }

  GRAPH_DUMP("Before optimizing linear nodes", graph);
  optimize_qlinear(graph);
  GRAPH_DUMP("Before embedding bags optimizations", graph);
  optimize_emb_quantize_per_tensor(graph);
  GRAPH_DUMP("Before fuse_embedding_bags", graph);
  fuse_embedding_bags(graph);
  GRAPH_DUMP("Before move_weights_to_memory", graph);
  move_weights_to_memory(graph);

  // Lint and DCE on the graph
  graph->lint();
  torch::jit::EliminateDeadCode(graph);

  // GRAPH_DUMP("Final Graph", graph);
}

} // namespace pace
