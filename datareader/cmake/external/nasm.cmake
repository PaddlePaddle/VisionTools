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

SET(NASM_SOURCES_DIR ${THIRD_PARTY_PATH}/source/nasm)
SET(NASM_INSTALL_DIR ${THIRD_PARTY_PATH}/nasm)

SET(NASM_BIN "${NASM_INSTALL_DIR}/bin/nasm" CACHE FILEPATH "nasm binary." FORCE)
SET(NASM_PKG "nasm-2.10.07.tar.bz2")
#SET(NASM_URL "https://www.nasm.us/pub/nasm/releasebuilds/2.10.07/${NASM_PKG}")
SET(NASM_URL "http://10.88.151.33:8070/${NASM_PKG}")

ExternalProject_Add(
    extern_nasm
    URL ${NASM_URL}
    PREFIX ${NASM_SOURCES_DIR}
    UPDATE_COMMAND  tar -jxvf ${NASM_SOURCES_DIR}/src/${NASM_PKG} 
    BINARY_DIR ${NASM_SOURCES_DIR}/src/extern_nasm
    CONFIGURE_COMMAND ./configure --prefix=${NASM_INSTALL_DIR}
    BUILD_COMMAND make
)

set(CMAKE_ASM_NASM_COMPILER "${NASM_INSTALL_DIR}/bin/nasm" CACHE FILEPATH "nasm binary" FORCE)
ADD_LIBRARY(nasm MODULE IMPORTED GLOBAL)
SET_PROPERTY(TARGET nasm PROPERTY IMPORTED_LOCATION ${NASM_BIN})
ADD_DEPENDENCIES(nasm extern_nasm)
