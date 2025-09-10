/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ATen/Parallel.h>
#include <ATen/quantized/Quantizer.h>
#include <ops/kernels/binary_kernels.h>
#include <utils/kernel_utils.h>
#include <utils/utils.h>
#include <cstdint>

namespace pace {

namespace kernels {

template <typename Taddend>
void qmul_add_kernel(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& addend,
    const at::Tensor& output) {
  // da is float (from emb), b is quantized, addend is quantized/float
  // output is quantized
  // 1. mul_out = (da * ((b - b zpoint) * b_scale)) or (da * (b * b scale -
  // (b zpoint * b scale)))
  // 2. add_out = mul_out + ((addend - addend zpoint) * addend scale) or
  // mul_out + (addend * addend scale - (addend zpoint *addend scale)) if addend
  // is quantized or add_out = mul_out + addend if addend is float
  // 3. final_out = (add_out * 1/output_scale) + output_zpoint

  // Multiplicant 1 - embedding bag output - (BS, x96)
  float* a_ptr = a.data_ptr<float>();

  // Multiplicant 2 - GEMM output - (BS, x96)
  uint8_t* b_ptr = b.data_ptr<uint8_t>();
  float b_scale = static_cast<float>(b.q_scale());
  int b_zpoint = static_cast<int>(b.q_zero_point());

  // Addend - Residual input - (BS, x96)
  Taddend* addend_ptr = addend.data_ptr<Taddend>();
  float addend_scale = 1;
  int addend_zpoint = 0;
  if (addend.scalar_type() == at::kQUInt8) {
    addend_scale = static_cast<float>(addend.q_scale());
    addend_zpoint = static_cast<int>(addend.q_zero_point());
  }

  // Output - (BS, x96)
  uint8_t* output_ptr = output.data_ptr<uint8_t>();
  float output_scale = 1 / static_cast<float>(output.q_scale());
  int output_zpoint = static_cast<int>(output.q_zero_point());

  int M = output.size(0);
  int N = output.size(1);

  at::parallel_for(0, M, 1, [&](int64_t start_idx, int64_t end_idx) {
    int chunk_size = end_idx - start_idx;
    int chunk_offset = start_idx * N;

    kernels::impl::qmul_add_kernel_mx96<Taddend>(
        a_ptr + chunk_offset,
        b_ptr + chunk_offset,
        b_scale,
        b_zpoint,
        addend_ptr + chunk_offset,
        addend_scale,
        addend_zpoint,
        output_ptr + chunk_offset,
        output_scale,
        output_zpoint,
        chunk_size,
        N);
  });
}

template void qmul_add_kernel<uint8_t>(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&);

template void qmul_add_kernel<float>(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&);

} // namespace kernels

} // namespace pace
