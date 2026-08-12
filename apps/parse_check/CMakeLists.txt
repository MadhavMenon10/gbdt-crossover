add_executable(parse_check src/parse_check_main.cpp)
target_link_libraries(parse_check PRIVATE core)
### Used to process CSV files
include(FetchContent)
FetchContent_Declare(rapidcsv GIT_REPOSITORY https://github.com/d99kris/rapidcsv.git GIT_TAG v8.97)
FetchContent_MakeAvailable(rapidcsv)
target_link_libraries(parse_check PRIVATE rapidcsv)
###
target_link_directories(parse_check PRIVATE /opt/homebrew/opt/llvm/lib/unwind /opt/homebrew/opt/llvm/lib/c++)
target_link_libraries(parse_check PRIVATE unwind)
target_link_options(parse_check PRIVATE -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++)
