add_executable(fil_bench src/fil_bench_main.cu)
target_link_libraries(fil_bench PRIVATE core)
target_include_directories(fil_bench PRIVATE include)

# Same rapidcsv dependency every other bench needs
include(FetchContent)
FetchContent_Declare(rapidcsv GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git GIT_TAG v8.97)
FetchContent_MakeAvailable(rapidcsv)
target_link_libraries(fil_bench PRIVATE rapidcsv)

# cuda_utils,
FetchContent_Declare(cuda_utils GIT_REPOSITORY https://github.com/MadhavMenon10/cuda-utils.git GIT_TAG v1.0)
FetchContent_MakeAvailable(cuda_utils)
add_library(cuda_utils_fil INTERFACE)  # separately named to avoid collision if gpu_bench's target is in the same build
target_include_directories(cuda_utils_fil INTERFACE ${cuda_utils_SOURCE_DIR})
target_link_libraries(fil_bench PRIVATE cuda_utils_fil)

# RAPIDS dependencies
find_package(Treelite CONFIG REQUIRED)
find_package(nvforest CONFIG REQUIRED)
find_package(raft CONFIG REQUIRED)
target_link_libraries(fil_bench PRIVATE nvforest::nvforest++ treelite::treelite raft::raft)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(fil_bench PRIVATE CUDA::cudart)
if(APPLE)
    target_link_directories(fil_bench PRIVATE /opt/homebrew/opt/llvm/lib/unwind /opt/homebrew/opt/llvm/lib/c++)
    target_link_libraries(fil_bench PRIVATE unwind)
    target_link_options(fil_bench PRIVATE -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++)
endif()
