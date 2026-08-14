add_executable(gpu_bench src/gpu_bench_main.cpp)
target_link_libraries(gpu_bench PRIVATE core)
### Used to process CSV files
include(FetchContent)
FetchContent_Declare(rapidcsv GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git GIT_TAG v8.97)
FetchContent_MakeAvailable(rapidcsv)
target_link_libraries(gpu_bench PRIVATE rapidcsv)
if (BUILD_CUDA)
    target_compile_definitions(gpu_bench PRIVATE BUILD_CUDA)
    target_link_libraries(gpu_bench PRIVATE cuda)
endif()
###

