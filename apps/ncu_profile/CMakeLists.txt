add_executable(ncu_profile src/ncu_profile_main.cpp)
target_link_libraries(ncu_profile PRIVATE core)
if (BUILD_CUDA)
    target_compile_definitions(ncu_profile PRIVATE BUILD_CUDA)
    target_link_libraries(ncu_profile PRIVATE cuda)
endif()


