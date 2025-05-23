cmake_minimum_required(VERSION 3.2)
project( FloorPlanGenerator LANGUAGES CXX CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# find_package(CUDA QUIET REQUIRED)
enable_language(CUDA)

# set( CMAKE_VERBOSE_MAKEFILE on )
set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# set(
#     CUDA_NVCC_FLAGS
#     ${CUDA_NVCC_FLAGS};
#     -O3
# )

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
# set(CUDA_NVCC_FLAGS -std=c++20  -Xcompiler -O3)

include (CTest)
include_directories(lib)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(cuda_headers "./cuda/combine.cuh" "./cuda/generate.cuh" "./cuda/helper.cuh"  "./cuda/common.cuh"  )
set(cuda_codes   "./cuda/combine.cu"  "./cuda/generate.cu")
set(headers "./lib/storage.h"   "./lib/iter.h"   "./lib/log.h"   "./lib/cvHelper.h"   "./lib/calculator.h"   "./lib/mpHelper.h"   "./lib/generate.h"   "./lib/combine.h"   "./lib/search.h"   "./lib/viewer.h"   "./lib/combinePostProcess.h"   "./cuda/process.h"   "./cuda/generateHandler.h"   "./cuda/combineHandler.h"   "./lib/globals.h") 
set(codes   "./src/storage.cpp" "./src/iter.cpp" "./src/log.cpp" "./src/cvHelper.cpp" "./src/calculator.cpp" "./src/mpHelper.cpp" "./src/generate.cpp" "./src/combine.cpp" "./src/search.cpp" "./src/viewer.cpp" "./src/combinePostProcess.cpp" "./cuda/process.cpp" "./cuda/generateHandler.cpp" "./cuda/combineHandler.cpp" "./src/main.cpp")

list(APPEND cxx_warning_flags "-Wall" "-Wextra")
# list(APPEND cuda_flags_lst "-G" "-Xptxas=-v")
list(APPEND cuda_flags_lst "-lineinfo" "-Xptxas=-v")
# list(APPEND cuda_flags_lst "-lineinfo" "-Xptxas=-O2 -v")
add_executable(FloorPlanGenerator ${cuda_headers} ${cuda_codes} ${headers} ${codes}) 

# target_compile_options(FloorPlanGenerator PRIVATE -Wall -Wextra)
target_compile_options(FloorPlanGenerator PRIVATE $<$<COMPILE_LANGUAGE:CXX>: ${cxx_warning_flags}> $<$<COMPILE_LANGUAGE:CUDA>: ${cuda_flags_lst}>)
target_link_libraries( FloorPlanGenerator PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} )