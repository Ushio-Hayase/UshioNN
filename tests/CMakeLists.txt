
# --- 유닛 테스트 빌드 ---
if (USHIONN_BUILD_TESTING)
    find_package(GTest)

    enable_testing() # CTest 사용 활성화

    # 테스트 실행 파일 이름
    set(UNIT_TEST_RUNNER_NAME nunet_tests)

    # 테스트 소스 파일 목록 (tests/ 디렉토리의 파일들을 추가)
    file(GLOB_RECURSE nunet_tests_src "*.cpp")

    add_executable(${UNIT_TEST_RUNNER_NAME} ${nunet_tests_src})

    # cuda 헤더 불러오기
    target_include_directories(${UNIT_TEST_RUNNER_NAME}
            PRIVATE
            ${CUDA_INCLUDE_DIRS}
    )

    target_link_libraries(${UNIT_TEST_RUNNER_NAME}
            PRIVATE
            nunet::lib
            GTest::gtest GTest::gtest_main
    )
    if (USE_ROCM)
        target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE USE_ROCM)
    elseif (USE_CUDA)
        target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE USE_CUDA)
        target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE TARGET_CUDA_ARCH=${CUDA_ARCHITECTURES})
    elseif (USE_CPU)
        target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE USE_CPU)
    else ()
        message(FATAL_ERROR "You haven't selected a device to use")
    endif ()

    if (USE_SIMD)
        target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE USE_SIMD)
        if (USE_AVX512)
            target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE USE_AVX512)
        endif ()
        if (USE_AVX2)
            target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE USE_AVX2)
        endif ()
        if (USE_AVX)
            target_compile_definitions(${UNIT_TEST_RUNNER_NAME} PRIVATE USE_AVX)
        endif ()
    endif ()

    # CTest에 테스트 추가
    include(GoogleTest)
    gtest_discover_tests(${UNIT_TEST_RUNNER_NAME})
endif ()
