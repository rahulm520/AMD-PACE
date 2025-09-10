/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#ifndef LIBXSMM_MLP_KERNEL_H
#define LIBXSMM_MLP_KERNEL_H

#include <ATen/ATen.h>
#include <ops/libxsmm_dependency/tensor_helper.h>

namespace pace {
namespace kernels {

// Dummy class for NoActivation template
class NoOpActivation {
 public:
  NoOpActivation(long BSb, long Hk, long K, long K2) {}

  template <typename T>
  void operator()(T* in, T* out) const {}
};

// Aliasing template names
using ReLUActivation = ReLUFwdTPP<at::BFloat16>;
using GeluActivation = GeluFwdTPP<at::BFloat16>;
using SiLUActivation = SiLUFwdTPP<at::BFloat16>;
using MulActivation = MulTPP<at::BFloat16, at::BFloat16>;

/**
 * @brief Performs a fully connected layer operation with multiplication.
 *
 * Performs matrix multiplication involving input, intermediate, and weight
 * tensors, optionally applying a bias and an activation function.
 *
 * @tparam ActivationTPP Type of the activation function to apply
 * @param t_in Input tensor
 * @param t_in1 Intermediate tensor
 * @param t_wt Weight tensor
 * @param t_bias Bias tensor
 * @param t_out Output tensor
 */
template <typename ActivationTPP>
void libxsmmlinear_kernel(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out);

} // namespace kernels
} // namespace pace

#endif // LIBXSMM_MLP_KERNEL_H
