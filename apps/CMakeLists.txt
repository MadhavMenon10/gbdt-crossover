add_subdirectory(parse_check)
if(BUILD_CUDA)
    add_subdirectory(gpu_bench)
endif()

