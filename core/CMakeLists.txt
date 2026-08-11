add_library(core STATIC src/tree.cpp src/model.cpp src/model_loader.cpp src/cpu_reference.cpp)
###
#Used to process JSON files using nlohmann/json as C++ does not provide native JSON support
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(json)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${json_SOURCE_DIR})
###
target_include_directories(core PUBLIC include)
target_compile_features(core PUBLIC cxx_std_23)
target_compile_options(core PRIVATE -Wall -Wextra -pedantic)
target_compile_definitions(core PUBLIC _LIBCPP_DISABLE_AVAILABILITY)
target_link_libraries(core PUBLIC nlohmann_json)
