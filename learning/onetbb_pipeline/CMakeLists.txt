# Copyright (c) 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.1)

project(main CXX)

# include(../../common/cmake/common.cmake)

# set_common_project_settings(tbb)
find_package(TBB REQUIRED)
# find_package(THREADS REQUIRED)
find_package(Threads REQUIRED)

add_executable(main main.cpp)

target_compile_definitions(main PUBLIC _CONSOLE)
target_link_libraries(main TBB::tbb Threads::Threads)
target_compile_options(main PRIVATE ${TBB_CXX_STD_FLAG})

set(EXECUTABLE "$<TARGET_FILE:main>")
# set(ARGS 2:4 -t 5)
# set(LIGHT_ARGS 1:2 -t 5)

# add_execution_target(run_main main ${EXECUTABLE} "${ARGS}")
# add_execution_target(light_test_main main ${EXECUTABLE} "${LIGHT_ARGS}")
