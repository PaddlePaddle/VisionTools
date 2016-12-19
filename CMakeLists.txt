cmake_minimum_required(VERSION 3.2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_SOURCE_DIR})

include(flags)

find_package(PythonLibs 2.7 REQUIRED)
find_package(PythonInterp 2.7 REQUIRED)
find_package(Glog REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${PROJ_ROOT})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${LIBGLOG_INCLUDE_DIR})
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

enable_testing()

add_subdirectory(transformer)
