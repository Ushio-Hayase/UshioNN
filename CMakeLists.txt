cmake_minimum_required(VERSION 3.21)
project(nunet LANGUAGES CXX)

set(NUNET_BUILD_TESTING OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_ROCM "Build for AMD ROCm using HIP" OFF)
option(USE_CUDA "Build for NVIDIA CUDA" OFF)
option(USE_CPU "Build for CPU" ON)

option(USE_AVX ON)
option(USE_AVX2 ON)
option(USE_AVX512 OFF)
option(USE_AVX10 OFF)


if(USE_CUDA)
    # CUDA 라이브러리 출력
    message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Include DIRS: ${CUDA_INCLUDE_DIRS}")
endif()

add_subdirectory(src)

add_subdirectory(tests)

message(STATUS "CMake configuration done. You can now build with 'cmake --build .'")

