
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

    # CTest에 테스트 추가
    include(GoogleTest)
    gtest_discover_tests(${UNIT_TEST_RUNNER_NAME})
endif ()
