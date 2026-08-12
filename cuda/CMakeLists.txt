enable_language(CUDA)
add_library(cuda STATIC src/)
target_include_directories(cuda PUBLIC include)
target_compile_features(cuda PUBLIC cxx_std_23)
