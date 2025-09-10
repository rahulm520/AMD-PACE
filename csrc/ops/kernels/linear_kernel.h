/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef LINEAR_KERNEL_H
#define LINEAR_KERNEL_H

#include <ATen/ATen.h>
#include <ops/jit_helper.h>
#include <pace_tensor/pace_tensor_impl.h>
#include <pace_tensor/quantize_utils.h>
#include <vector>

namespace pace {

namespace kernels {

/**
 * @brief Wraps the matmul primitive with the correct arguments
 *
 * @tparam with_reorder If the weights and biases should be reordered or not
 * @param input A matrix
 * @param weight B matrix
 * @param bias Bias to be added to AxB
 * @param output C matrix
 * @param attr Any post-ops and extra attributes to be used
 * @param q_scales All the scale values
 * @param post_ops_mem Any post-op memory if required (eg: binary)
 */
void linear_kernel(
    const memory& input,
    PACETensor& weight,
    PACETensor& bias,
    memory& output,
    primitive_attr& attr,
    bool with_reorder = true,
    const c10::optional<quantize_scales>& q_scales = {c10::nullopt},
    const c10::optional<quantize_zeropoint>& q_zp = {c10::nullopt},
    const std::vector<memory>& post_ops_mem = {});

} // namespace kernels

} // namespace pace

#endif // LINEAR_KERNEL_H
