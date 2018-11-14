#!/usr/bin/python

""" setup file for datareader
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
use_turbojpeg = True
if os.environ.get('USE_TURBO_JPEG', '') == '0':
    use_turbojpeg = False

third_libs = os.environ.get('THIRD_LIBS_INSTALL_PATH', 'third_party')
if third_libs == 'third_party':
    third_libs = os.path.join(curfolder, third_libs)

opencvincludedir = os.path.join(third_libs, 'opencv/include')
opencvlibdir = os.path.join(third_libs, 'opencv/lib')
jpegturboincludedir = os.path.join(third_libs, 'turbojpeg/include')
jpegturbolibdir = os.path.join(third_libs, 'turbojpeg/lib')

pythonlibdir = os.environ.get('PYTHON_LIBRARIES', '')
if pythonlibdir == '':
    pythonlibdir = os.path.join(third_libs, 'python27/lib')
else:
    if os.path.isfile(pythonlibdir):
        pythonlibdir = os.path.dirname(pythonlibdir)

opencvlibs = [
    "opencv_world", "IlmImf", "ittnotify", "libjasper",
    "libpng", "libtiff", "libwebp", "zlib"
]

if not use_turbojpeg:
    opencvlibs += ["libjpeg"] #include it for libturbojpeg.a has also included this
    linklibs = [os.path.join(opencvlibdir, 'lib' + lib + '.a') for lib in opencvlibs]
else:
    linklibs = [os.path.join(opencvlibdir, 'lib' + lib + '.a') for lib in opencvlibs]
    linklibs += [os.path.join(jpegturbolibdir, 'libturbojpeg.a')]

#use xlinker instead of libraries, to avoid link error
extralinkflag = ["-fopenmp"] + ['-Xlinker', "-("] + linklibs \
          + ['-Xlinker', "-)"] \
          + [r"-Wl,--rpath=$ORIGIN", "-Wl,--rpath=$ORIGIN/../so", "-Wl,-z,origin"]

def make_transform_ext(name, pyxfile, ext_root):
    """ make python extension about libpytransform
    """
    srcfiles = [pyxfile] + glob.glob(os.path.join(ext_root, 'src', '*.cpp'))

    macros = [('NDEBUG', '1')]
    if use_turbojpeg:
        macros += [('USETURBOJPEG', None)]
    return Extension(
        name=name,
        sources=srcfiles,
        define_macros=macros,
        extra_compile_args=["-O3", "-msse4.2", "-fopenmp", "-std=c++11", "-fPIC"],
        extra_link_args= extralinkflag,
        include_dirs=[
            ext_root + '/include', 
            ext_root + '/src',
            opencvincludedir,
            jpegturboincludedir
        ],
        library_dirs=[opencvlibdir, pythonlibdir, jpegturbolibdir],
        language='c++')


modname = 'datareader'
version = '0.0.1'

#make cpp extension
lib_modname = modname + '.transformer.libpytransform'
pyxfile = os.path.join(cpp_root, 'libpytransform.pyx')
extensions = [make_transform_ext(lib_modname, pyxfile, cpp_root)]

#build package of datareader
pysource = 'python'
setup(name=modname,
    version=version,
    description="a package for data loading and preprocessing in training model",
    packages=find_packages(where=pysource),
    package_dir={'': pysource},
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions)
