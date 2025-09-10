/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 *******************************************************************************/

#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdexcept>

#include <core/threading.h>

namespace pace {

void thread_bind(const std::vector<int32_t>& cpu_core_list) {
  omp_set_num_threads(cpu_core_list.size());

#pragma omp parallel num_threads(cpu_core_list.size())
  {
    int thread_id = omp_get_thread_num();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core_list[thread_id], &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) !=
        0) {
      throw std::runtime_error("Fail to bind cores.");
    }
  }
}

} // namespace pace
