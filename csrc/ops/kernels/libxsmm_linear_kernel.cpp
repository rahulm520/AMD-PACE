/******************************************************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 ******************************************************************************************************************/

/******************************************************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.
 *
 * For information on the license, see the LICENSE file. Further information:
 * https://github.com/libxsmm/tpp-pytorch-extension/
 * Source Code:
 * https://github.com/libxsmm/tpp-pytorch-extension/blob/mlperf_infer_31/src/csrc/llm/fused_llm_infer.cpp
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************************************************/

/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************************************************/

#include <ops/kernels/libxsmm_linear_kernel.h>
#include <ops/libxsmm_dependency/threaded_loops.h>
#include <ops/libxsmm_dependency/utils.h>
#include <ops/libxsmm_dependency/xsmm_functors.h>
namespace pace {

namespace kernels {
static int PACE_LARGE_CACHE_OPT = false;
static int PACE_NCB_BLOCK_SIZE = env2int("PACE_NCB_BLOCK_SIZE", 64);
static int PACE_FT_OPT_SIZE = env2int("PACE_NCB_BLOCK_SIZE", 256);
static const char* PACE_GEMM_LOOP_SCHEME =
    getenv("PACE_GEMM_LOOP_SCHEME") ? getenv("PACE_GEMM_LOOP_SCHEME") : "aCB";
static const char* PACE_GEMM_LOOP_SCHEME_1 =
    getenv("PACE_GEMM_LOOP_SCHEME") ? getenv("PACE_GEMM_LOOP_SCHEME") : "aCB";

template <typename ActivationTPP>
void apply_activation(
    ActivationTPP& activation_tpp,
    at::BFloat16* in1,
    at::BFloat16* out) {
  if constexpr (std::is_same_v<ActivationTPP, MulActivation>) {
    activation_tpp(in1, out, out);
  } else {
    activation_tpp(out, out);
  }
}

template <typename ActivationTPP>
void libxsmmlinear_kernel(
    at::Tensor& t_in,
    at::Tensor& t_in1,
    at::Tensor& t_wt,
    at::Tensor& t_bias,
    at::Tensor& t_out) {
  auto in_sizes = t_in.sizes();
  auto BS = in_sizes[0] * in_sizes[1];
  auto wt_sizes = t_wt.sizes();
  auto C = in_sizes[2];
  using DataType = at::BFloat16;
  auto Nc = wt_sizes[1];
  auto Hc = C / Nc;
  auto Nk = wt_sizes[0];
  auto Hk = wt_sizes[3];
  auto K = Nk * Hk;
  auto t_wt_V = wt_tensor_for_fwd(Nk, Hk, Nc, Hc, t_wt);
  bool with_bias = (t_bias.dim() > 0);
  auto in = GetVLAPtr<DataType>(t_in, {Nc, Hc});
  auto in1 = GetVLAPtr<DataType>(t_in1, {Nk, Hk});
  auto wt_V = GetVLAPtr<DataType>(t_wt_V, {Nc, Hc * Hk});
  auto bias = GetVLAPtr<DataType>(t_bias, {Hk});
  auto out = GetVLAPtr<DataType>(t_out, {Nk, Hk});

  auto Ncb = Nc;
  auto BSb = 64L;
  auto rem = BS % 64;
  if (PACE_LARGE_CACHE_OPT)
    Ncb = PACE_NCB_BLOCK_SIZE;

  // Activation TPP initialization
  ActivationTPP activation_tpp(BSb, Hk, K, K);
  ActivationTPP activation_tpp_rem(rem, Hk, K, K);

  auto copy_bias_tpp = CpyBiasTPP<DataType>(BSb, Hk, K);
  auto copy_bias_tpp_rem = CpyBiasTPP<DataType>(rem, Hk, K);
  auto zero_tpp = SetZeroTPP<DataType>(BSb, Hk, K);
  auto zero_tpp_rem = SetZeroTPP<DataType>(rem, Hk, K);
  auto brgemm_tpp = BrgemmTPP<at::BFloat16, DataType>(
      BSb, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb);
  auto brgemm_tpp_rem = BrgemmTPP<at::BFloat16, DataType>(
      rem, Hk, Hc, Hc, Hk * Hc, C, Hk, K, 1.0, 0, Ncb);

  {
    auto loop_scheme = PACE_GEMM_LOOP_SCHEME_1;
    auto igemm_loop =
        ThreadedLoop<3>({{0, Nc, Ncb, false}, {0, BS, BSb}, {Nk}}, loop_scheme);
    igemm_loop([&](int* ind) {
      int nc = ind[0], s1 = ind[1], nk = ind[2];
      auto count = nc + Ncb < Nc ? Ncb : Nc - nc;
      bool is_rem = (s1 + BSb > BS);

      if (!is_rem) {
        if (nc == 0) {
          if (with_bias) {
            copy_bias_tpp(bias[nk], out[s1][nk]);
          } else {
            zero_tpp(out[s1][nk]);
          }
        }
        brgemm_tpp(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, true);

        if (!(nc + Ncb < Nc)) {
          apply_activation<ActivationTPP>(
              activation_tpp, in1[s1][nk], out[s1][nk]);
        }
      } else {
        if (nc == 0) {
          if (with_bias) {
            copy_bias_tpp_rem(bias[nk], out[s1][nk]);
          } else {
            zero_tpp_rem(out[s1][nk]);
          }
        }
        brgemm_tpp_rem(in[s1][nc], wt_V[nk][nc], out[s1][nk], count, false);

        if (!(nc + Ncb < Nc)) {
          apply_activation<ActivationTPP>(
              activation_tpp_rem, in1[s1][nk], out[s1][nk]);
        }
      }
    });
  }
}
// // Instantiate libxsmmlinear_kernel for each activation type
#define INSTANTIATE_LIBXSMM_KERNEL(ActivationType)    \
  template void libxsmmlinear_kernel<ActivationType>( \
      at::Tensor & t_in,                              \
      at::Tensor & t_in1,                             \
      at::Tensor & t_wt,                              \
      at::Tensor & t_bias,                            \
      at::Tensor & t_out);

INSTANTIATE_LIBXSMM_KERNEL(ReLUActivation)
INSTANTIATE_LIBXSMM_KERNEL(GeluActivation)
INSTANTIATE_LIBXSMM_KERNEL(SiLUActivation)
INSTANTIATE_LIBXSMM_KERNEL(MulActivation)
INSTANTIATE_LIBXSMM_KERNEL(NoOpActivation)

#undef INSTANTIATE_LIBXSMM_KERNEL

} // namespace kernels

} // namespace pace
