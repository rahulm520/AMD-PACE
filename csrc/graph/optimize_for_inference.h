/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef OPTIMIZE_FOR_INFERENCE_H
#define OPTIMIZE_FOR_INFERENCE_H

#include <vector>

#include <torch/csrc/jit/ir/ir.h>

namespace pace {

/**
 * @brief Entry point to the torch script pass manager graph pass
 *
 * @param graph Graph to be optimized
 */
void Optimize(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace pace

#endif // OPTIMIZE_FOR_INFERENCE_H
