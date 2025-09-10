/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef OPTIMIZE_EMBEDDING_BAGS_H
#define OPTIMIZE_EMBEDDING_BAGS_H

#include <torch/csrc/jit/ir/ir.h>
#include <utils/graph_utils.h>

#include <vector>

namespace pace {

void optimize_emb_quantize_per_tensor(std::shared_ptr<torch::jit::Graph> graph);

} // namespace pace

#endif // OPTIMIZE_EMBEDDING_BAGS_H
