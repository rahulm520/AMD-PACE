/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 *******************************************************************************/

#ifndef REGISTER_H
#define REGISTER_H

#include <graph/optimize_for_inference.h>
#include <torch/csrc/jit/passes/pass_manager.h>

struct TORCH_API PACEPassOptimize
    : public torch::jit::PassManager<PACEPassOptimize> {
  static void enable_pace_fusion(bool enabled) {
    if (enabled) {
      registerPass(pace::Optimize);
    } else {
      clearPass();
    }
  }

  // override PassManager::registerPass to register pre-pass
  static void registerPass(torch::jit::GraphPass p) {
    if (!isRegistered()) {
      // passID(registerPrePass(std::move(p)), true);
      passID(registerPostPass(std::move(p)), true);
      isRegistered(true);
    }
  }

  // override PassManager::clearPass to clear pre-pass
  static void clearPass() {
    if (isRegistered()) {
      torch::jit::clearPrePass(passID());
      isRegistered(true);
    }
  }
};

#endif // REGISTER_H
