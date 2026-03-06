cmake_minimum_required(VERSION 3.21)
project(nunet LANGUAGES CXX)

set(NUNET_BUILD_TESTING OFF)

set(CMAKE_CXX_STANDARD 20)

option(USE_ROCM "Build for AMD ROCm using HIP" OFF)
option(USE_CUDA "Build for NVIDIA CUDA" OFF)
option(USE_CPU "Build for CPU" ON)

option(USE_SIMD "Use SIMD" ON)
option(USE_AVX "Build with AVX" ON)
option(USE_AVX2 "Build with AVX2" ON)
option(USE_AVX512 "Bulid with AVX512" ON)


if(USE_AVX512)
    if(MSVC)
        add_compile_options(/arch:AVX512)
    else()
        add_compile_options(-mavx512f -mavx512vl -mavx512bw -mavx512dq)
    endif()
elseif(USE_AVX2)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2)
    endif()
elseif(USE_AVX)
    if(MSVC)
        add_compile_options(/arch:AVX)
    else()
        add_compile_options(-mavx)
    endif()
endif()

add_subdirectory(src)


if(USE_ROCM)
    target_compile_definitions(nunet_lib PRIVATE USE_ROCM)
elseif(USE_CUDA)
    target_compile_definitions(nunet_lib PRIVATE USE_CUDA)
elseif(USE_CPU)
    target_compile_definitions(nunet_lib PRIVATE USE_CPU)
else()
    message(FATAL_ERROR "You haven't selected a device to use")
endif()

if(USE_SIMD)
    target_compile_definitions(nunet_lib PRIVATE USE_SIMD)
endif()

if(USE_AVX512)
    target_compile_definitions(nunet_lib PRIVATE USE_AVX512)
endif()
if(USE_AVX2)
    target_compile_definitions(nunet_lib PRIVATE USE_AVX2)
endif()
if(USE_AVX)
    target_compile_definitions(nunet_lib PRIVATE USE_AVX)
endif()


if(USE_CUDA)
    # CUDA 라이브러리 출력
    message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Include DIRS: ${CUDA_INCLUDE_DIRS}")
endif()

add_subdirectory(tests)

message(STATUS "CMake configuration done. You can now build with 'cmake --build .'")

