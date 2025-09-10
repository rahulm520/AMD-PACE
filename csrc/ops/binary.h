/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef BINARY_H
#define BINARY_H

#include <ATen/ATen.h>

namespace pace {

/**
 * @brief Implementation for quantized mul + add implementation
 * for the DLRMv2 model interaction layer post-ops
 *
 * @param input0 Tensor Multiplicant
 * @param input1 Tensor Multiplier
 * @param addend Tensor Addend
 * @param output_scale Output scale for INT8 output
 * @param output_zpoint Output zero point for INT8 output
 * @param output_dtype Output dtype
 * @return at::Tensor
 */
at::Tensor qmul_add(
    const at::Tensor& input0,
    const at::Tensor& input1,
    const at::Tensor& addend,
    const at::Scalar& output_scale,
    const at::Scalar& output_zpoint,
    const at::ScalarType& output_dtype);

} // namespace pace

#endif // BINARY_H
