
# --- 유닛 테스트 빌드 ---
if(NUNET_BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)

    enable_testing() # CTest 사용 활성화

    # 테스트 실행 파일 이름
    set(UNIT_TEST_RUNNER_NAME ushionn_tests)

    # 테스트 소스 파일 목록 (tests/ 디렉토리의 파일들을 추가)
    file(GLOB_RECURSE ushionn_tests_src "*.cpp")

    add_executable(${UNIT_TEST_RUNNER_NAME} ${ushionn_tests_src})

    # cuda 헤더 불러오기
    target_include_directories(${UNIT_TEST_RUNNER_NAME}
        PRIVATE
        ${CUDA_INCLUDE_DIRS}
    )

    target_link_libraries(${UNIT_TEST_RUNNER_NAME}
        PRIVATE
        ushionn::lib # 우리가 빌드한 라이브러리
        GTest::gtest GTest::gtest_main # Google Test (main 함수 포함) 또는 gtest
        # 만약 gtest_main 대신 gtest를 사용하면, tests/main.cpp 에 직접 GTest 초기화 코드 작성 필요
    )

    # CTest에 테스트 추가
    include(GoogleTest)
    gtest_discover_tests(${UNIT_TEST_RUNNER_NAME})
endif()
