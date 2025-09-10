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
 * https://github.com/libxsmm/tpp-pytorch-extension/blob/mlperf_infer_31/src/csrc/utils.h
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************************************************/

/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************************************************/

#ifndef _TPP_UTILS_H_
#define _TPP_UTILS_H_

#include <cstdlib>

inline int guess_mpi_rank() {
  const char* env_names[] = {
      "RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};
  static int guessed_rank = -1;
  if (guessed_rank >= 0)
    return guessed_rank;
  for (int i = 0; i < 4; i++) {
    if (getenv(env_names[i]) != NULL) {
      int r = atoi(getenv(env_names[i]));
      if (r >= 0) {
        printf("My guessed rank = %d\n", r);
        guessed_rank = r;
        return guessed_rank;
      }
    }
  }
  guessed_rank = 0;
  return guessed_rank;
}

inline int env2int(const char* env_name, int def_val = 0) {
  int val = def_val;
  auto env = getenv(env_name);
  if (env)
    val = atoi(env);

  return val;
}

#endif //_TPP_UTILS_H_
