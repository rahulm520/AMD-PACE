/******************************************************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************************************************/

/******************************************************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.
 *
 * For information on the license, see the LICENSE file. Further information:
 * https://github.com/libxsmm/tpp-pytorch-extension/
 * Source Code:
 * https://github.com/libxsmm/tpp-pytorch-extension/blob/mlperf_infer_31/src/csrc/tensor_helper.h
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************************************************/

/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************************************************/

#ifndef _TENSOR_HELPER_H_
#define _TENSOR_HELPER_H_

#include "utils.h"
#include "vla.h"
#include "xsmm_functors.h"
using namespace tpp;

enum PassType { OTH, FWD, BWD, UPD };
extern PassType globalPass;

template <typename T>
inline at::Tensor wt_tensor_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  const int BS = get_vnni_block_size<T>();
#if 0
  TPP_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  return input.view({Nk, Nc, Hc/BS, BS, Hk}).permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto Hcp2 = (Hc + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hcp2, Hk, BS});
  auto out = GetVLAPtr<T>(output, {Hcp2 * Hk * BS});
  auto in = GetVLAPtr<T>(input, {Hc * Hk});
  auto n2v_tpp = XformExtTPP<T>(Hc, Hk, Hcp2 * BS, Hk, XformTPP::XFORM_N2V_TPP);
//   RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

inline at::Tensor wt_tensor_for_fwd(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  // RECORD_SCOPE(w_vnni, {input});
  if (input.dtype() != at::kFloat) {
    if (input.dim() == 5) {
      return input;
    } else {
      if (input.dtype() == at::kBFloat16) {
        return wt_tensor_n2v<bfloat16>(Nk, Hk, Nc, Hc, input);
      }
      // else if (input.dtype() == at::kBFloat8) {
      //   return wt_tensor_n2v<bfloat8>(Nk, Hk, Nc, Hc, input);
      // }

      else {
        TPP_ASSERT(false, "Unsupported datatype!");
      }
    }
  } else {
    return input;
  }
}

#endif // _TENSOR_HELPER_H_
