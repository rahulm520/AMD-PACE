/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef FUSE_EMBEDDING_BAGS_H
#define FUSE_EMBEDDING_BAGS_H

#include <torch/csrc/jit/ir/ir.h>
#include <vector>

namespace pace {

/**
 * @brief Find and fuse embedding bags which are followed by concat
 *
 * @param graph : Torchscript graph
 */
void fuse_embedding_bags(std::shared_ptr<torch::jit::Graph> graph);

} // namespace pace

#endif // FUSE_EMBEDDING_BAGS_H
