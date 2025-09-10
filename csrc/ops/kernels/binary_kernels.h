/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/ATen.h>

namespace pace {

namespace kernels {

/**
 * @brief This function performs the quantized elementwise multiplication and
 * addition operation (purely fused). The output is quantized to the output
 * scale and zero point.
 *
 * @tparam Taddend
 * @param da The input matrix A
 * @param qb The input matrix B
 * @param qaddend The input matrix C
 * @param output_scale The output scale
 * @param output_zpoint The output zero point
 * @param output_dtype The output data type
 * @return at::Tensor
 */
template <typename Taddend>
void qmul_add_kernel(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& addend,
    const at::Tensor& output);

namespace impl {

/**
 * @brief This function implements intrinsics for the quantized elementwise
 * multiplication and addition operation (purely fused). This function is
 * optimized for the case when the number of columns in matrix A is 96.
 * The output is quantized to the output scale and zero point.
 *
 * @tparam Taddend
 * @param a The input matrix A
 * @param b The input matrix B
 * @param b_scale The input scale of matrix B
 * @param b_zpoint The input zero point of matrix B
 * @param c The input matrix C
 * @param c_scale The input scale of matrix C
 * @param c_zpoint The input zero point of matrix C
 * @param output The output matrix
 * @param o_scale The output scale
 * @param o_zpoint The output zero point
 * @param M The number of rows in matrix A
 * @param N The number of columns in matrix B
 */
template <typename Taddend>
void qmul_add_kernel_mx96(
    float* a,
    uint8_t* b,
    float b_scale,
    int b_zpoint,
    Taddend* c,
    float c_scale,
    int c_zpoint,
    uint8_t* output,
    float o_scale,
    int o_zpoint,
    int M,
    int N);
} // namespace impl

} // namespace kernels

} // namespace pace
