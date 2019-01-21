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

SET(LUACV_SOURCES_DIR ${THIRD_PARTY_PATH}/source/luacv)
SET(LUACV_INSTALL_DIR ${THIRD_PARTY_PATH}/luacv)
SET(LUACV_INCLUDE_DIR "${LUACV_INSTALL_DIR}/include" CACHE PATH "luacv include directory." FORCE)

SET(LUACV_LIBRARIES "${LUACV_INSTALL_DIR}/lib/libluacv.a" CACHE FILEPATH "luacv library." FORCE)
SET(LUACV_REPOSITORY "https://github.com/walloollaw/luaOpenCV.git")
SET(LUACV_TAG "d9b7a47c3a6373020c22223c59bcd57534aa8633")

INCLUDE_DIRECTORIES(${LUACV_INCLUDE_DIR})

if(NOT OPENCV_INSTALL_DIR)
    message(SEND_ERROR "not found installed opencv in ${OPENCV_INSTALL_DIR}")
endif()

if(NOT LUA_INSTALL_DIR)
    message(SEND_ERROR "not found installed lua in ${LUA_INSTALL_DIR}")
endif()

ExternalProject_Add(
    extern_luacv
    DEPENDS opencv lua
    GIT_REPOSITORY  ${LUACV_REPOSITORY}
    GIT_TAG         ${LUACV_TAG}
    PREFIX          ${LUACV_SOURCES_DIR}
    UPDATE_COMMAND  cp ${LUACV_SOURCES_DIR}/src/extern_luacv/src/raw_bind_generated.inc.full ${LUACV_SOURCES_DIR}/src/extern_luacv/src/raw_bind_generated.inc
    CMAKE_ARGS      -DTHIRD_PARTY_PATH=${THIRD_PARTY_PATH}
                    -DLUA_INSTALL_DIR=${LUA_INSTALL_DIR}
                    -DOPENCV_INSTALL_DIR=${OPENCV_INSTALL_DIR}
    CMAKE_CACHE_ARGS -DTHIRD_PARTY_PATH:PATH=${THIRD_PARTY_PATH}
)

ADD_LIBRARY(luacv STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET luacv PROPERTY IMPORTED_LOCATION ${LUACV_LIBRARIES})
ADD_DEPENDENCIES(luacv extern_luacv lua opencv)
#LINK_LIBRARIES(luacv lua)

LIST(APPEND external_project_dependencies luacv)

#INSTALL(DIRECTORY ${LUACV_INCLUDE_DIR} DESTINATION third_party/luacv)
#INSTALL(FILES ${LUACV_LIBRARIES} DESTINATION third_party/luacv/lib)
