/***********************************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 **********************************************************************************************/

/***********************************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.
 *
 * For information on the license, see the LICENSE file.
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/
 * Source Code:
 *https://github.com/libxsmm/tpp-pytorch-extension/blob/mlperf_infer_31/src/csrc/common_loops.cpp
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ***********************************************************************************************
 * Author: Dhiraj Kalamkar (Intel Corp.)
 **********************************************************************************************/

#include <ops/libxsmm_dependency/threaded_loops.h>
#include <functional>
#include <string>
#include <unordered_map>

void par_nested_loops_A(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for nowait
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      int ind[1] = {a0};
      body_func(ind);
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_AB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(2) nowait
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_aB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for nowait
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_bA(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
         b0 += loopSpecs[1].step) {
#pragma omp for nowait
      for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
           a0 += loopSpecs[0].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_Ba(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for nowait
    for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
         b0 += loopSpecs[1].step) {
      for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
           a0 += loopSpecs[0].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_BA(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(2) nowait
    for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
         b0 += loopSpecs[1].step) {
      for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
           a0 += loopSpecs[0].step) {
        int ind[2] = {a0, b0};
        body_func(ind);
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_ABC(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
#pragma omp for collapse(3) nowait
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
             c0 += loopSpecs[2].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_aBC(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for collapse(2) nowait
      for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
           b0 += loopSpecs[1].step) {
        for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
             c0 += loopSpecs[2].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_acB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
      for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
           c0 += loopSpecs[2].step) {
#pragma omp for nowait
        for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
             b0 += loopSpecs[1].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_aCb(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for nowait
      for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
           c0 += loopSpecs[2].step) {
        for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
             b0 += loopSpecs[1].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

void par_nested_loops_aCB(
    LoopSpecs* loopSpecs,
    std::function<void(int*)> body_func,
    std::function<void()> init_func,
    std::function<void()> fini_func) {
#pragma omp parallel
  {
    if (init_func)
      init_func();
    for (int a0 = loopSpecs[0].start; a0 < loopSpecs[0].end;
         a0 += loopSpecs[0].step) {
#pragma omp for collapse(2) nowait
      for (int c0 = loopSpecs[2].start; c0 < loopSpecs[2].end;
           c0 += loopSpecs[2].step) {
        for (int b0 = loopSpecs[1].start; b0 < loopSpecs[1].end;
             b0 += loopSpecs[1].step) {
          int ind[3] = {a0, b0, c0};
          body_func(ind);
        }
      }
    }
    if (fini_func)
      fini_func();
  }
}

std::unordered_map<std::string, par_loop_kernel> pre_defined_loops = {
    {"A", par_nested_loops_A},
    {"AB", par_nested_loops_AB},
    {"BA", par_nested_loops_BA},
    {"bA", par_nested_loops_bA},
    {"Ba", par_nested_loops_Ba},
    {"aB", par_nested_loops_aB},
    {"ABC", par_nested_loops_ABC},
    {"aBC", par_nested_loops_aBC},
    {"acB", par_nested_loops_acB},
    {"aCb", par_nested_loops_aCb},
    {"aCB", par_nested_loops_aCB},
};
