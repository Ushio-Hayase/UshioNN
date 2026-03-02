cmake_minimum_required(VERSION 3.20)
project(ushionn LANGUAGES CXX CUDA)

set(USHIONN_BUILD_TESTING ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# CUDA 라이브러리 출력
message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA Include DIRS: ${CUDA_INCLUDE_DIRS}")

add_subdirectory(src)

add_subdirectory(tests)

message(STATUS "CMake configuration done. You can now build with 'cmake --build .'")

