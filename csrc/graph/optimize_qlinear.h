/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef OPTIMIZE_QLINEAR_H
#define OPTIMIZE_QLINEAR_H

#include <torch/csrc/jit/ir/ir.h>
#include <vector>

namespace pace {

void optimize_qlinear(std::shared_ptr<torch::jit::Graph> graph);
} // namespace pace

#endif // OPTIMIZE_QLINEAR_H
