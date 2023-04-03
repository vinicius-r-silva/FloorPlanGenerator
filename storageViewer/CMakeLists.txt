cmake_minimum_required(VERSION 3.2)
project( FloorPlanGenerator LANGUAGES CXX)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# set( CMAKE_VERBOSE_MAKEFILE on )
set(CMAKE_CXX_STANDARD 20)


# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(codes   "viewer.cpp")

list(APPEND cxx_warning_flags "-Wall" "-Wextra")
add_executable(FloorPlanGenerator ${codes}) 

target_compile_options(FloorPlanGenerator PRIVATE -Wall -Wextra)
target_link_libraries( FloorPlanGenerator PUBLIC ${OpenCV_LIBS} )