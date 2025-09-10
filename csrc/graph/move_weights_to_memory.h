/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef MOVE_WEIGHTS_TO_MEMORY
#define MOVE_WEIGHTS_TO_MEMORY

#include <torch/csrc/jit/ir/ir.h>
#include <vector>

namespace pace {

/**
 * @brief Moves weights and biases for particular nodes PACETensors
 *
 * @param graph Graph in which the nodes with weights and biases are to be found
 */
void move_weights_to_memory(std::shared_ptr<torch::jit::Graph> graph);

} // namespace pace

#endif // MOVE_WEIGHTS_TO_MEMORY
