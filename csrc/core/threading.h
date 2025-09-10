/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved. Notified per clause 4(b) of the license.
 * Portions of this file consist of AI-generated content
 *******************************************************************************/

#include <vector>

namespace pace {

/**
 * @brief Accepts a list of cores and binds the current process
 * to the gives cores
 *
 * @param cpu_core_list Lists of cores for the process to be binded
 * @return true
 * @return false
 */
void thread_bind(const std::vector<int32_t>& cpu_core_list);

} // namespace pace
