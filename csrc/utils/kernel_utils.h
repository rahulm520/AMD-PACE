/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include <cpuinfo.h>

namespace pace {

namespace kernels {

inline bool has_avx512_support() {
  return (
      cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() &&
      cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512vl());
}

inline bool has_avx512vnni_support() {
  return (cpuinfo_has_x86_avx512vnni());
}

} // namespace kernels

} // namespace pace

#endif
