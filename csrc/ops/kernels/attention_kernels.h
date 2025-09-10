/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/
#ifndef PACE_ATTENTION_KERNEL_H
#define PACE_ATTENTION_KERNEL_H

#include <ops/jit_helper.h>

namespace pace {

namespace kernels {

/**
 * @brief SDPA kernel implementation with 4 different approaches
 * SDPA supported for F32 and BF16 input data types
 *
 * @param input_Q_mem
 * @param input_K_mem
 * @param input_V_mem
 * @param output_mem
 * @param input_mask_mem
 * @param use_KQ
 */
void attention_kernel(
    const memory& input_Q_mem,
    const memory& input_K_mem,
    const memory& input_V_mem,
    memory& output_mem,
    const memory& input_mask_mem,
    const int use_KQ);

} // namespace kernels

} // namespace pace

#endif // PACE_ATTENTION_KERNEL_H
