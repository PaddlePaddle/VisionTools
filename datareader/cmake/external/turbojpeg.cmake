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
# limitations under the License.

INCLUDE(ExternalProject)

SET(TURBO_JPEG_SOURCES_DIR ${THIRD_PARTY_PATH}/source/turbojpeg)
SET(TURBO_JPEG_INSTALL_DIR ${THIRD_PARTY_PATH}/turbojpeg)
SET(TURBO_JPEG_INCLUDE_DIR "${TURBO_JPEG_INSTALL_DIR}/include" CACHE PATH "turbojpeg include directory." FORCE)
SET(TURBO_JPEG_LIB_DIR "${TURBO_JPEG_INSTALL_DIR}/lib" CACHE PATH "turbojpeg lib directory." FORCE)

SET(TURBO_JPEG_LIBRARIES "${TURBO_JPEG_LIB_DIR}/libturbojpeg.a" CACHE FILEPATH "turbojpeg library." FORCE)
SET(TURBO_JPEG_REPOSITORY "https://github.com/libjpeg-turbo/libjpeg-turbo.git")
SET(TURBO_JPEG_TAG "2.0.0")

SET(HEADER_ROOT "${TURBO_JPEG_SOURCES_DIR}/src/extern_turbojpeg")
SET(TURBO_JPEG_HEADERS
    jconfig.h
    ${HEADER_ROOT}/jerror.h
    ${HEADER_ROOT}/jmorecfg.h
    ${HEADER_ROOT}/jpeglib.h
    ${HEADER_ROOT}/turbojpeg.h)

INCLUDE_DIRECTORIES(${TURBO_JPEG_INCLUDE_DIR})

include(external/nasm)
if (NOT DEFINED CMAKE_ASM_NASM_COMPILER)
    message(FATAL_ERROR "not found nasm")
else()
    message("set nasm to ${CMAKE_ASM_NASM_COMPILER}")
endif()

SET(CMAKE_CFLAGS " -fPIC -fvisibility=hidden ")
ExternalProject_Add(
    extern_turbojpeg
    DEPENDS gflags nasm
    GIT_REPOSITORY  ${TURBO_JPEG_REPOSITORY}
    GIT_TAG         ${TURBO_JPEG_TAG}
    PREFIX          ${TURBO_JPEG_SOURCES_DIR}
    UPDATE_COMMAND  sed -i "s/#define DLLEXPORT$/#define DLLEXPORT __attribute__((visibility(\"default\")))/g" ${TURBO_JPEG_SOURCES_DIR}/src/extern_turbojpeg/turbojpeg.h
    BUILD_COMMAND   make
    INSTALL_COMMAND mkdir -p ${TURBO_JPEG_LIB_DIR} && mkdir -p ${TURBO_JPEG_INCLUDE_DIR} && cp libturbojpeg.a ${TURBO_JPEG_LIBRARIES} && cp ${TURBO_JPEG_HEADERS} ${TURBO_JPEG_INCLUDE_DIR}
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_C_FLAGS=${CMAKE_CFLAGS}
                    -DCMAKE_INSTALL_PREFIX=${TURBO_JPEG_INSTALL_DIR}
                    -DCMAKE_INSTALL_LIBDIR=${TURBO_JPEG_INSTALL_DIR}/lib
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    -DCMAKE_ASM_NASM_COMPILER=${CMAKE_ASM_NASM_COMPILER}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DWITH_JPEG8=ON
                    -DENABLE_SHARED=OFF
                    -DENABLE_STATIC=ON
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${TURBO_JPEG_INSTALL_DIR}
                     -DCMAKE_INSTALL_LIBDIR:PATH=${TURBO_JPEG_INSTALL_DIR}/lib
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)

ADD_LIBRARY(turbojpeg STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET turbojpeg PROPERTY IMPORTED_LOCATION ${TURBO_JPEG_LIBRARIES})
ADD_DEPENDENCIES(turbojpeg extern_turbojpeg gflags nasm)
#LINK_LIBRARIES(glog gflags)

LIST(APPEND external_project_dependencies turbojpeg)

IF(WITH_C_API)
  INSTALL(DIRECTORY ${TURBO_JPEG_INCLUDE_DIR} DESTINATION third_party/turbojpeg)
  INSTALL(FILES ${TURBO_JPEG_LIBRARIES} DESTINATION third_party/turbojpeg/lib)
ENDIF()
