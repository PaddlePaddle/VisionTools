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

SET(OPENCV_SOURCES_DIR ${THIRD_PARTY_PATH}/source/opencv)
SET(OPENCV_INSTALL_DIR ${THIRD_PARTY_PATH}/opencv)
SET(OPENCV_INCLUDE_DIR "${OPENCV_INSTALL_DIR}/include" CACHE PATH "opencv include directory." FORCE)

SET(OPENCV_LIBRARIES "${OPENCV_INSTALL_DIR}/lib" CACHE FILEPATH "opencv libraries" FORCE)

SET(OPENCV_REPOSITORY "https://github.com/opencv/opencv.git")
SET(OPENCV_TAG "3.4.1")

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})

ExternalProject_Add(
    extern_opencv 
    GIT_REPOSITORY  ${OPENCV_REPOSITORY}
    GIT_TAG         ${OPENCV_TAG}
    PREFIX          ${OPENCV_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
                    -DCMAKE_INSTALL_LIBDIR=${OPENCV_INSTALL_DIR}/lib
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                    -DBUILD_JASPER:BOOL=ON
                    -DBUILD_JPEG:BOOL=ON
                    -DBUILD_OPENEXR:BOOL=ON
                    -DBUILD_PNG:BOOL=ON
                    -DBUILD_TIFF:BOOL=ON
                    -DBUILD_SHARED_LIBS:BOOL=OFF
                    -DBUILD_ZLIB:BOOL=ON
                    -DBUILD_PERF_TESTS=OFF
                    -DWITH_1394:BOOL=OFF
                    -DWITH_FFMPEG:BOOL=OFF
                    -DWITH_EIGEN:BOOL=ON
                    -DWITH_GSTREAMER:BOOL=OFF
                    -DWITH_GTK:BOOL=OFF
                    -DWITH_UNICAP:BOOL=OFF
                    -DWITH_V4L:BOOL=OFF
                    -DWITH_XIMEA:BOOL=OFF
                    -DWITH_XINE:BOOL=OFF
                    -DBUILD_TBB:BOOL=OFF
                    -DWITH_PVAPI=OFF
                    -DWITH_CUDA:BOOL=OFF
                    -DWITH_PROTOBUF:BOOL=OFF
                    -DOPENCV_ENABLE_NONFREE:BOOL=ON
                    -DBUILD_opencv_world:BOOL=ON
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${OPENCV_INSTALL_DIR}
                     -DCMAKE_INSTALL_LIBDIR:PATH=${OPENCV_INSTALL_DIR}/lib
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)

add_custom_command(
    TARGET extern_opencv
    POST_BUILD
    COMMAND cp ${OPENCV_INSTALL_DIR}/share/OpenCV/3rdparty${OPENCV_LIBRARIES}/lib*.a ${OPENCV_LIBRARIES}
    COMMENT "copy thirdparty libs to 'opencv/lib'"
)

ADD_LIBRARY(opencv STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET opencv PROPERTY IMPORTED_LOCATION ${OPENCV_LIBRARIES})
ADD_DEPENDENCIES(opencv extern_opencv)

LIST(APPEND external_project_dependencies opencv)
