add_subdirectory(parse_check)
add_subdirectory(cpu_bench)
if(BUILD_CUDA)
    add_subdirectory(gpu_bench)
endif()

