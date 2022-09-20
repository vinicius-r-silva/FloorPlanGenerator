cmake_minimum_required(VERSION 2.8)
project( main )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include (CTest)
include_directories(lib)

set(CMAKE_BUILD_TYPE Debug)
set(headers "./lib/storage.h") 
set(codes "./src/main.cpp" "./src/storage.cpp")
add_executable(main ${headers} ${codes}) 
# add_executable( storage src/storage.cpp )
target_link_libraries( main PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} )