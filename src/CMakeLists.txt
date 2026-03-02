file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "*.cpp" "*.cu")
foreach(TMP_PATH ${SOURCES_FILES})
    foreach(EXCLUDE_PATH ${EXCLUDE_PATHS})
        string(FIND ${TMP_PATH} ${EXCLUDE_PATH} RESULT)
        if(NOT ${RESULT} EQUAL -1)
            list(REMOVE_ITEM ushonn_src ${TMP_PATH})
        endif()
    endforeach(EXCLUDE_PATH)
endforeach(TMP_PATH)

if(USE_ROCM)
    enable_language(CXX HIP)

    # 윈도우 환경 변수 HIP_PATH를 기반으로 hipify-clang.exe 경로 탐색
    find_program(HIPIFY_CLANG NAMES hipify-clang
        PATHS "$ENV{HIP_PATH}/bin"
        REQUIRED)
    if(NOT HIPIFY_PERL)
        message(FATAL_ERROR "hipify-perl not found. Please ensure ROCm is installed and in your PATH.")
    endif()

    set(HIP_SOURCES "")

    # 각 .cu 파일에 대해 변환 규칙 생성
    foreach(cu_file ${SOURCE_FILES})
        get_filename_component(abs_cu_file ${cu_file} ABSOLUTE)
        get_filename_component(base_name ${cu_file} NAME_WE)

        # 빌드 디렉토리 내에 생성될 .hip 파일 경로
        set(hip_file "${CMAKE_CURRENT_BINARY_DIR}/${base_name}.hip")

        # 윈도우에서는 의존성 문제 없는 hipify-clang.exe 사용
        add_custom_command(
            OUTPUT ${hip_file}
            COMMAND ${HIPIFY_CLANG} ${abs_cu_file} -o ${hip_file}
            DEPENDS ${abs_cu_file}
            COMMENT "Converting ${cu_file} to HIP using hipify-clang.exe"
        )

        list(APPEND HIP_SOURCES ${hip_file})
    endforeach()

    # 변환된 HIP 소스로 실행 파일 생성
    add_library(ushionn_lib STATIC ${HIP_SOURCES})

    if(MSVC)
        target_compile_options(ushoinn_lib PRIVATE /EHsc)
    endif()

else()
    enable_language(CXX CUDA)
    add_library(ushionn_lib STATIC ${SOURCE_FILES})
endif()



add_library(ushionn::lib ALIAS ushionn_lib)

target_include_directories(ushionn_lib PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${CUDA_INCLUDE_DIRS})


target_link_libraries(ushionn_lib
    PUBLIC
    ${CUDA_LIBRARIES}
)
