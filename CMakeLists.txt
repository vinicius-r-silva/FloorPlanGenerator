cmake_minimum_required (VERSION 3.8)

project ("FloorPlanGenerator")

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_subdirectory ("FloorPlanGenerator")
# add_subdirectory ("docs")