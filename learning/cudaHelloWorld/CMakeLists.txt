### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 3.2)
project(test LANGUAGES CXX CUDA)
# find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
# set(
#     CUDA_NVCC_FLAGS
#     ${CUDA_NVCC_FLAGS};
#     -O3 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_86,code=sm_86
# )

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

# message(STATUS "${INSTALLED_GPU_CCS_1}")
# message(STATUS "${INSTALLED_GPU_CCS_2}")
# message(STATUS "${INSTALLED_GPU_CCS_3}")
# message(STATUS "${CUDA_ARCH_LIST}")

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_BUILD_TYPE Release)

# cuda_select_nvcc_arch_flags(gpu auto)

# For compilation ...
# Specify target & source files to compile it from
# cuda_add_executable(
#     gpu
#     gpu.cu
#     gpu.h
#     cpu.cpp
# )

add_executable(
    gpu
    gpu.cu
    gpu.h
    cpu.cpp
)

target_compile_options(gpu PRIVATE -Wall -Wextra -Wpedantic)

# add_executable(
#     cpu
# )

# For linking ...
# Specify target & libraries to link it with
# target_link_libraries(
#     gpu
#     )

    