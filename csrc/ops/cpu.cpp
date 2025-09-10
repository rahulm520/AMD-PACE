/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <ops/cpu.h>

namespace pace {
engine& cpu_eng() {
  static engine cpu_eng_(engine::kind::cpu, 0);
  return cpu_eng_;
}
} // namespace pace
