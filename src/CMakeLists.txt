file(GLOB_RECURSE ushionn_src CONFIGURE_DEPENDS "*.cpp" "*.cu")
foreach(TMP_PATH ${thread_pool_SOURCES})
    foreach(EXCLUDE_PATH ${EXCLUDE_PATHS})
        string(FIND ${TMP_PATH} ${EXCLUDE_PATH} RESULT)
        if(NOT ${RESULT} EQUAL -1)
            list(REMOVE_ITEM ushonn_src ${TMP_PATH})
        endif()
    endforeach(EXCLUDE_PATH)
endforeach(TMP_PATH)

add_library(ushionn_lib STATIC ${ushionn_src})
add_library(ushionn::lib ALIAS ushionn_lib)

target_include_directories(ushionn_lib PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${CUDA_INCLUDE_DIRS})


target_link_libraries(ushionn_lib
    PUBLIC
    ${CUDA_LIBRARIES}
)
