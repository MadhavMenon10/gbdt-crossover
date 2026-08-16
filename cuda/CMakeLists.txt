enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
add_library(cuda STATIC src/tree_traversal_device.cu src/launch_tree_traversal_device.cu src/device_model.cu)
#Used to bring in my CUDA Utils file
include(FetchContent)
FetchContent_Declare(cuda_utils GIT_REPOSITORY https://github.com/MadhavMenon10/cuda-utils.git GIT_TAG v1.0)
FetchContent_MakeAvailable(cuda_utils)
add_library(cuda_utils INTERFACE)
target_include_directories(cuda_utils INTERFACE ${cuda_utils_SOURCE_DIR})
###
target_link_libraries(cuda PRIVATE cuda_utils)
target_link_libraries(cuda PUBLIC CUDA::cudart)
target_include_directories(cuda PUBLIC include)
target_include_directories(cuda PUBLIC ${CMAKE_SOURCE_DIR}/core/include)
target_compile_features(cuda PUBLIC cxx_std_23)

