add_executable(cpu_bench src/cpu_bench_main.cpp)
target_link_libraries(cpu_bench PRIVATE core)
# Used for XGBoost
include(FetchContent)
FetchContent_Declare(rapidcsv GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git GIT_TAG v8.97)
FetchContent_Declare(xgboost_c_api URL https://raw.githubusercontent.com/dmlc/xgboost/v3.4.0/include/xgboost/c_api.h DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(rapidcsv)
FetchContent_MakeAvailable(xgboost_c_api)
target_link_libraries(cpu_bench PRIVATE rapidcsv)
add_library(xgboost_api INTERFACE)
target_include_directories(xgboost_api INTERFACE ${xgboost_c_api_SOURCE_DIR})
target_link_libraries(cpu_bench PRIVATE xgboost_api)
# Locates the real compiled library sitting inside the pip-installed package 
# We can't hardcode a path, since it differs between the M1 and Runpod
execute_process(COMMAND python3 -c "import xgboost, os; print(os.path.dirname(xgboost.__file__))" OUTPUT_VARIABLE XGBOOST_PYTHON_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_library(XGBOOST_LIB NAMES xgboost PATHS "${XGBOOST_PYTHON_DIR}/lib" NO_DEFAULT_PATH)
target_link_libraries(cpu_bench PRIVATE ${XGBOOST_LIB})
