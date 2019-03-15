#!/bin/bash

#
# script used by cmake to build wheel
#

ROOT=$(readlink -f `dirname ${BASH_SOURCE}[0]`)
BUILD_TYPE=$1
python ${ROOT}/setup.py ${BUILD_TYPE}
ret=$?
if [[ $ret -ne 0 ]];then
    echo "failed to build wheel"
    exit $ret
fi

if [[ $BUILD_TYPE = "develop" ]];then
    echo "succeed to build module for developing"
    exit 0
fi

if [[ -n $WHEEL_DIST_DIR ]];then
    rm -rf ${WHEEL_DIST_DIR}/dist
    mv dist ${WHEEL_DIST_DIR}
else
    WHEEL_DIST_DIR=${ROOT}
fi

echo "succeed generate wheel in ${WHEEL_DIST_DIR}/dist"
exit 0
