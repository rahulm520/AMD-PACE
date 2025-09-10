# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

set(ENABLE_CLANG_FORMAT $ENV{ENABLE_CLANG_FORMAT})

if(ENABLE_CLANG_FORMAT)
    find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format)

    if(NOT CLANG_FORMAT_EXECUTABLE)
        message(WARNING "clang-format not found! Disabling clang-format.")
        set(ENABLE_CLANG_FORMAT OFF)
    endif()
endif()

# Download the format file from pytorch if not available
set(CLANG_FORMAT_FILE "${CMAKE_SOURCE_DIR}/.clang-format")
if(NOT EXISTS ${CLANG_FORMAT_FILE})

    set(CLANG_FORMAT_URL "https://raw.githubusercontent.com/pytorch/pytorch/refs/tags/v2.4.1/.clang-format")
    file(DOWNLOAD ${CLANG_FORMAT_URL} ${CLANG_FORMAT_FILE}
        TIMEOUT 60  # Time-out in seconds
        STATUS download_status)

    list(GET download_status 0 download_status_code)
    if(NOT download_status_code EQUAL 0)
        message(WARNING "Failed to download file for clang-format: ${download_status}, Disabling clang-format")
    endif()
endif()
