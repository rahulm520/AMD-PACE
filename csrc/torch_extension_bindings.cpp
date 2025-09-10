/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 ******************************************************************************/

#include <core/logging.h>
#include <core/threading.h>
#include <graph/register.h>
#include <torch_extension_bindings.h>

void torch_extension_bindings(pybind11::module& m) {
  m.def("thread_bind", &pace::thread_bind);

  m.def("enable_pace_fusion", &PACEPassOptimize::enable_pace_fusion);

  m.def("pace_logger", &pace::pace_logger);
}
