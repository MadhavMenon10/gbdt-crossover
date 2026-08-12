cmake_minimum_required(VERSION 3.20)
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
endif()
project(gbdt-crossover LANGUAGES CXX)
add_subdirectory(core)
add_subdirectory(apps)


