/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef PACE_MLP_H
#define PACE_MLP_H

#include <ATen/ATen.h>

namespace pace {

/**
 * @brief Create a mlp block of a transformer network using the IMBPS
 * implementation with weights already split
 *
 * @param src Src tensor to the first matmul opearation
 * @param weight Array of weight tensors to the first matmul opearation
 * @param bias Array of Bias tensors to be added to first matmul operation
 * @param weights2 Array of weights tensors of the second matmul operation
 * @param bias2 Bias tensor to the second matmul operation
 * @param nlf Non linearity function
 * @param weights_gateProj Array of weight tensors to the gate projection
 * operation
 * @param bias_gateProj Array of Bias tensors to be added to gate projection
 * operation
 */
at::Tensor mlp_mlp_fusion(
    const at::Tensor& src,
    const std::vector<at::Tensor>& weights1,
    const c10::optional<std::vector<at::Tensor>>& bias1,
    const std::vector<at::Tensor>& weights2,
    const c10::optional<at::Tensor>& bias2,
    std::string nlf,
    const c10::optional<std::vector<at::Tensor>>& weights_gateProj,
    const c10::optional<std::vector<at::Tensor>>& bias_gateProj);

} // namespace pace

#endif // PACE_MLP_H
