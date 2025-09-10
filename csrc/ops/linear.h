/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef LINEAR_H
#define LINEAR_H

#include <ATen/ATen.h>

#include <ops/jit_helper.h>
#include <pace_tensor/pace_aten_interface.h>
#include <pace_tensor/pace_tensor_impl.h>

namespace pace {

/**
 * @brief Get the bias if available
 *
 * @param bias The bias Tensor to be parsed
 * @return PACETensor
 */
inline PACETensor get_bias_if_available(const at::Tensor& bias) {
  memory bias_mem = memory({}, cpu_eng(), JIT_MEMORY_NONE);
  if (bias.dim() != 0) {
    bias_mem = view_tensor_as_memory(
        bias, /*transpose*/ transpose_dims(), /*ndims*/ 2);
  }
  return PACETensor(bias_mem, {}, {});
}

/**
 * @brief fp32 -> fp32 and bf16 -> bf16 kernel for 2D Matrix Multiplication
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @return at::Tensor
 */
at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt);

/**
 * @brief fp32 -> fp32 and bf16 -> bf16 kernel for 2D Matrix Multiplication
 * with post-ops: Relu
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @return at::Tensor
 */
at::Tensor linear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt);

/**
 * @brief int8 -> int8 or int8 -> fp32 kernel for 2D Matrix Multiplication
 * int8 -> fp32 conversion happen properly only if output_scale = 1,
 * output_zpoint = 0 and output_dtype = kFloat
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @param output_scale Output scale to be used for requantization
 * @param output_zpoint Output zero point to be used for requantization
 * @param output_dtype Output dtype
 * @return at::Tensor
 */
at::Tensor qlinear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Scalar& output_scale,
    const at::Scalar& output_zpoint,
    const at::ScalarType& output_dtype);

/**
 * @brief int8 -> int8 or int8 -> fp32 kernel for 2D Matrix Multiplication
 * with post-ops: Relu
 * int8 -> fp32 conversion happen properly only if output_scale = 1,
 * output_zpoint = 0 and output_dtype = kFloat
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @param output_scale Output scale to be used for requantization
 * @param output_zpoint Output zero point to be used for requantization
 * @param output_dtype Output dtype
 * @return at::Tensor
 */
at::Tensor qlinear_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Scalar& output_scale,
    const at::Scalar& output_zpoint,
    const at::ScalarType& output_dtype);

/**
 * @brief int8 -> fp32 kernel for 2D Matrix Multiplication
 *  with post-ops: binary mul+ binaryadd
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @param multiplier Tensor multiplicant
 * @param addend Tensor addend
 * @param alpha alpha value for addend
 * @return at::Tensor
 */
at::Tensor qlinear_mul_add(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& multiplier,
    const at::Tensor& addend,
    const at::Scalar& alpha);

/**
 * @brief int8 -> fp32 kernel for 2D Matrix Multiplication
 * with post-ops: sigmoid/logistic
 *
 * @param input A matrix
 * @param weight B matrix
 * @param bias_opt Optional bias
 * @return at::Tensor
 */
at::Tensor qlinear_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt);

} // namespace pace

#endif // LINEAR_H
