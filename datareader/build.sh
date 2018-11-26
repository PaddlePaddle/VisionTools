#!/bin/bash

#
# script used by cmake to build wheel
#

ROOT=$(readlink -f `dirname ${BASH_SOURCE}[0]`)
if [[ $# -eq 2 ]];then
    export THIRD_LIBS_INSTALL_PATH=$1
    export BUILD_JPEG_TURBO=$2
fi

python ${ROOT}/setup.py bdist_wheel
ret=$?
if [[ $ret -ne 0 ]];then
    echo "failed to build wheel"
    exit $ret
fi

if [[ -n $WHEEL_DIST_DIR ]];then
    rm -rf ${WHEEL_DIST_DIR}/dist
    mv dist ${WHEEL_DIST_DIR}
else
    WHEEL_DIST_DIR=${ROOT}
fi

echo "succeed generate wheel in ${WHEEL_DIST_DIR}/dist"
exit 0
