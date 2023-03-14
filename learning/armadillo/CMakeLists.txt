
cmake_minimum_required(VERSION 3.2)
project(foo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
add_executable(foo main.cpp)
target_link_libraries(foo ${ARMADILLO_LIBRARIES})
target_compile_options(foo PRIVATE -Wall -Wextra)