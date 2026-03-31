cmake_minimum_required(VERSION 3.21)
project(ushionn LANGUAGES CXX)

set(USHIONN_BUILD_TESTING ON)

set(CMAKE_CXX_STANDARD 20)

option(USE_ROCM "Build for AMD ROCm using HIP" OFF)
option(USE_CUDA "Build for NVIDIA CUDA" ON)
option(USE_CPU "Build for CPU" ON)

set(CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_ARCHITECTURES 120)

option(USE_SIMD "Use SIMD" ON)
option(USE_AVX "Build with AVX" ON)
option(USE_AVX2 "Build with AVX2" ON)
option(USE_AVX512 "Build with AVX512" ON)

add_subdirectory(src)

if (USE_AVX512)
    if (MSVC)
        target_compile_options(nunet_lib PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX512>
        )
    else ()
        target_compile_options(nunet_lib PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512fp16>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx512f,-mavx512vl,-mavx512bw,-mavx512dq -mavx512fp16>
        )
    endif ()
elseif (USE_AVX2)
    if (MSVC)
        target_compile_options(nunet_lib PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>
        )
    else ()
        target_compile_options(nunet_lib PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2>
        )
    endif ()
elseif (USE_AVX)
    if (MSVC)
        target_compile_options(nunet_lib PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX>
        )
    else ()
        target_compile_options(nunet_lib PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx>
        )
    endif ()
endif ()

if (USE_ROCM)
    target_compile_definitions(nunet_lib PRIVATE USE_ROCM)
elseif (USE_CUDA)
    target_compile_definitions(nunet_lib PRIVATE USE_CUDA)
    target_compile_definitions(nunet_lib PRIVATE TARGET_CUDA_ARCH=${CUDA_ARCHITECTURES})
elseif (USE_CPU)
    target_compile_definitions(nunet_lib PRIVATE USE_CPU)
else ()
    message(FATAL_ERROR "You haven't selected a device to use")
endif ()

if (USE_SIMD)
    target_compile_definitions(nunet_lib PRIVATE USE_SIMD)
    if (USE_AVX512)
        target_compile_definitions(nunet_lib PRIVATE USE_AVX512)
    endif ()
    if (USE_AVX2)
        target_compile_definitions(nunet_lib PRIVATE USE_AVX2)
    endif ()
    if (USE_AVX)
        target_compile_definitions(nunet_lib PRIVATE USE_AVX)
    endif ()
endif ()

if (USE_CUDA)
    message(STATUS "Found CUDA: ${CUDAToolkit_LIBRARY_ROOT}")
    message(STATUS "CUDA Libraries Directory:  ${CUDAToolkit_LIBRARY_DIR}")
    message(STATUS "CUDA Include Directory: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA Architectures: ${CUDA_ARCHITECTURES}")
endif ()

add_subdirectory(tests)

message(STATUS "CMake configuration done. You can now build with 'cmake --build .'")

