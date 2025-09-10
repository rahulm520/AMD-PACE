/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_ATTENTION_H
#define PACE_ATTENTION_H

#include <ATen/ATen.h>

namespace pace {

/**
 * @brief Implementation for SDPA attention
 * Performs the ops: softmax ( ( add_mask( Q.K') ) ) . V
 * SDPA supported for F32 and BF16 input data types
 *
 * @param input_Q Tensor Q
 * @param input_K Tensor K
 * @param input_V Tensor V
 * @param input_mask Tensor mask
 * @param use_KQ Scalar Flag
 */
at::Tensor attention(
    const at::Tensor& input_Q,
    const at::Tensor& input_K,
    const at::Tensor& input_V,
    const c10::optional<at::Tensor>& input_mask,
    const c10::optional<at::Scalar>& use_KQ);

} // namespace pace

#endif // PACE_ATTENTION_H
