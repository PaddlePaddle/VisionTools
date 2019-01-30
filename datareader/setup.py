#!/usr/bin/python
"""
# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# function:
#   setup file for datareader
"""

import os
import sys
import glob
from setuptools import setup
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

print("### build datareader wheel with args:%s" % (' '.join(sys.argv)))
curfolder = os.path.dirname(os.path.abspath(__file__))
cpp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp')
with_turbo = False
if os.environ.get('WITH_TURBOJPEG', '') == '1':
    with_turbo = True

with_lua = True
if os.environ.get('WITH_LUA', '') == '0':
    with_lua = False

third_lib_root = os.environ.get('THIRD_LIBS_INSTALL_PATH', 'third_party')
if third_lib_root == 'third_party':
    third_lib_root = os.path.join(curfolder, third_lib_root)


def prepare_depdent_libs(third_lib_root, with_turbo, with_lua):
    """ prepare dependent libraries
    """
    include_dirs = []
    link_dirs = []
    link_libs = []

    include_dirs += [os.path.join(third_lib_root, 'opencv/include')]
    link_dirs += [os.path.join(third_lib_root, 'opencv/lib')]
    include_dirs += [os.path.join(third_lib_root, 'turbojpeg/include')]
    link_dirs += [os.path.join(third_lib_root, 'turbojpeg/lib')]

    opencv_install_dir = os.path.join(third_lib_root, 'opencv')
    opencvlibs = [
        "opencv_world", "IlmImf", "ittnotify", "libjasper", "libpng",
        "libtiff", "libwebp", "zlib"
    ]
    if with_turbo:
        link_libs = [
            os.path.join(third_lib_root, 'opencv/lib', 'lib' + lib + '.a')
            for lib in opencvlibs
        ]
        link_libs += [
            os.path.join(third_lib_root, 'turbojpeg/lib/libturbojpeg.a')
        ]
    else:
        opencvlibs += ["libjpeg"]
        link_libs += [
            os.path.join(third_lib_root, 'opencv/lib', 'lib' + lib + '.a')
            for lib in opencvlibs
        ]

    if with_lua:
        include_dirs += [os.path.join(third_lib_root, 'lua/include')]
        link_dirs += [os.path.join(third_lib_root, 'lua/lib')]
        link_libs += [os.path.join(third_lib_root, 'lua/lib/liblua.a')]

        include_dirs += [os.path.join(third_lib_root, 'luacv/include')]
        link_dirs += [os.path.join(third_lib_root, 'luacv/lib')]
        link_libs += [os.path.join(third_lib_root, 'luacv/lib/libluacv.a')]

    pythonlibdir = os.environ.get('PYTHON_LIBRARIES', '')
    if pythonlibdir == '':
        pythonlibdir = os.path.join(third_lib_root, 'python27/lib')
    else:
        if os.path.isfile(pythonlibdir):
            pythonlibdir = os.path.dirname(pythonlibdir)
    link_dirs += [pythonlibdir]

    return include_dirs, link_dirs, link_libs


def make_transform_ext(name, pyxfile, ext_root):
    """ make python extension about libpytransform
    """
    macros = [('NDEBUG', '1')]
    srcfiles = [pyxfile] + glob.glob(os.path.join(ext_root, 'src', '*.cpp'))
    if with_lua:
        macros += [('WITH_LUA', None)]
    else:
        srcfiles = filter(lambda n: n.find('lua') < 0, srcfiles)

    if with_turbo:
        macros += [('WITH_TURBOJPEG', None)]

    include_dirs, link_dirs, link_libs = prepare_depdent_libs(
        third_lib_root, with_turbo, with_lua)
    include_dirs += [ext_root, ext_root + '/include']
    extra_link_flag = ["-fopenmp"] + ['-Xlinker', "-("] + link_libs \
              + ['-Xlinker', "-)"] \
              + [r"-Wl,--rpath=$ORIGIN", "-Wl,--rpath=$ORIGIN/../so", "-Wl,-z,origin"]
    return Extension(
        name=name,
        sources=srcfiles,
        define_macros=macros,
        extra_compile_args=[
            "-O3", "-msse4.2", "-fopenmp", "-std=c++11", "-fPIC"
        ],
        extra_link_args=extra_link_flag,
        include_dirs=include_dirs,
        library_dirs=link_dirs,
        language='c++')


modname = 'visreader'
version = '1.0.0'

#make cpp extension
lib_modname = modname + '.transformer.libpytransform'
pyxfile = os.path.join(cpp_root, 'libpytransform.pyx')
extensions = [make_transform_ext(lib_modname, pyxfile, cpp_root)]

#build package of visreader
pysource = 'python'
setup(
    name=modname,
    version=version,
    url='https://github.com/PaddlePaddle/VisionTools.git',
    description="a package for data loading and preprocessing in training model",
    packages=find_packages(where=pysource),
    package_dir={'': pysource},
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions)
