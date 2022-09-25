cmake_minimum_required(VERSION 2.8)
project( FloorPlanGenerator )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include (CTest)
include_directories(lib)


set(CMAKE_BUILD_TYPE Debug)
set(headers "./lib/storage.h"   "./lib/iter.h"   "./lib/log.h"   "./lib/globals.h") 
set(codes   "./src/storage.cpp" "./src/iter.cpp" "./src/log.cpp" "./src/main.cpp")
add_executable(FloorPlanGenerator ${headers} ${codes}) 
target_compile_options(FloorPlanGenerator PRIVATE -Wall -Wextra -Wpedantic)
# add_executable( storage src/storage.cpp )
target_link_libraries( FloorPlanGenerator PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} )