# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

option(WITH_TURBOJPEG "Build with libturbojpeg" OFF)
option(WITH_LUA "Build with lua support" ON)
option(DEVELOP_MODE "Build python module in develop mode" OFF)

set(MODULE_BUILD_TYPE "bdist_wheel")
if (DEVELOP_MODE)
    SET(MODULE_BUILD_TYPE "develop")
endif()

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")

include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/opencv)    # download, build, install opencv

if (WITH_LUA)
    message("build with lua support")
    include(external/lua)       # download, build, install lua
    include(external/luacv)     # download, build, install luacv
endif()

if (WITH_TURBOJPEG)
    message("build with turbojpeg support")
    include(external/turbojpeg) # download, build, install turbojpeg
endif()

find_package(PythonLibs 2.7 REQUIRED)
if (PYTHONLIBS_FOUND)
    message("python libs: ${PYTHON_LIBRARIES}")
    message("python inc: ${PYTHON_INCLUDE_DIRS}")
else()
    message("not found python package")
endif()

message("dependent libs:${external_project_dependencies}")
message("python module build type: ${MODULE_BUILD_TYPE}")
add_custom_target(
    datareader_wheel
    ALL COMMAND env
        THIRD_LIBS_INSTALL_PATH=${THIRD_PARTY_PATH}
        WITH_TURBOJPEG=${WITH_TURBOJPEG}
        WITH_LUA=${WITH_LUA}
        PYTHON_LIBRARIES=${PYTHON_LIBRARIES}
        WHEEL_DIST_DIR=${CMAKE_CURRENT_BINARY_DIR}
        ${DATAREADER_SOURCE_DIR}/build.sh ${MODULE_BUILD_TYPE}
    WORKING_DIRECTORY ${DATAREADER_SOURCE_DIR}
    DEPENDS ${external_project_dependencies}
)
