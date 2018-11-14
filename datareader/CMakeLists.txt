# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(DATAREADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DATAREADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

project(datareader)

option(BUILD_TURBO_JPEG "Build with libturbojpeg" ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")

include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/opencv)      # download, build, install opencv

if (BUILD_TURBO_JPEG)
    include(external/turbojpeg) # download, build, install turbojpeg
endif()

set(THIRD_LIBS_INSTALL_PATH "${THIRD_PARTY_PATH}")
set(USE_TURBO_JPEG "${BUILD_JPEG_TURBO}")
set(WHL_BUILD_CMD "python ${DATAREADER_SOURCE_DIR}/setup.py build bdist_wheel")

find_package(PythonLibs 2.7 REQUIRED)
if (PYTHONLIBS_FOUND)
    message("python libs: ${PYTHON_LIBRARIES}")
    message("python inc: ${PYTHON_INCLUDE_DIRS}")
else()
    message("not found python package")
endif()

message("build wheel ${WHL_BUILD_CMD}")
message("dependent libs:${external_project_dependencies}")
add_custom_target(
    datareader_wheel
    ALL COMMAND env
        THIRD_LIBS_INSTALL_PATH=${THIRD_LIBS_INSTALL_PATH}
        BUILD_JPEG_TURBO=${BUILD_JPEG_TURBO}
        PYTHON_LIBRARIES=${PYTHON_LIBRARIES}
        WHEEL_DIST_DIR=${CMAKE_CURRENT_BINARY_DIR}
        ${DATAREADER_SOURCE_DIR}/build.sh 
    WORKING_DIRECTORY ${DATAREADER_SOURCE_DIR}
    DEPENDS ${external_project_dependencies}
)
