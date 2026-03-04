cmake_minimum_required(VERSION 3.21)
project(ushionn LANGUAGES CXX CUDA)

set(USHIONN_BUILD_TESTING ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_ROCM "Build for AMD ROCm using HIP" OFF)
option(USE_CUDA "Build for NVIDIA CUDA" ON)
option(USE_CPU "Build for CPU" OFF)

# CUDA 라이브러리 출력
message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA Include DIRS: ${CUDA_INCLUDE_DIRS}")

add_subdirectory(src)

add_subdirectory(tests)

message(STATUS "CMake configuration done. You can now build with 'cmake --build .'")

