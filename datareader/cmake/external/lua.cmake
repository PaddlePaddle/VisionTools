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

SET(LUA_SOURCES_DIR ${THIRD_PARTY_PATH}/source/lua)
SET(LUA_INSTALL_DIR ${THIRD_PARTY_PATH}/lua)

SET(LUA_BIN "${LUA_INSTALL_DIR}/bin/lua" CACHE FILEPATH "lua binary." FORCE)
SET(LUA_PKG "lua-5.3.5.tar.gz")
SET(LUA_URL "http://www.lua.org/ftp/${LUA_PKG}")
#SET(LUA_URL "http://10.88.151.33:8070/${LUA_PKG}")

SET(LUA_INCLUDE_DIR "${LUA_INSTALL_DIR}/include" CACHE PATH "lua include directory." FORCE)
set(LUA_LIBRARIES "${LUA_INSTALL_DIR}/lib/liblua.a" CACHE FILEPATH "LUA_LIBRARIES" FORCE)

INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})

ExternalProject_Add(
    extern_lua
    URL ${LUA_URL}
    PREFIX ${LUA_SOURCES_DIR}
    UPDATE_COMMAND  tar -zxvf ${LUA_SOURCES_DIR}/src/${LUA_PKG} 
    BINARY_DIR ${LUA_SOURCES_DIR}/src/extern_lua
    CONFIGURE_COMMAND ""
    #CONFIGURE_COMMAND pwd && ./configure --prefix=${LUA_INSTALL_DIR}
    BUILD_COMMAND make linux CFLAGS=-fPIC LDFLAGS=-lncurses INSTALL_TOP=${LUA_INSTALL_DIR}
    INSTALL_COMMAND make linux install LDFLAGS=-lncurses INSTALL_TOP=${LUA_INSTALL_DIR}
)

ADD_LIBRARY(lua STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET lua PROPERTY IMPORTED_LOCATION ${LUA_LIBRARIES})
ADD_DEPENDENCIES(lua extern_lua)

LIST(APPEND external_project_dependencies lua)
