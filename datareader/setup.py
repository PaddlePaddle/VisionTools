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


if '--turbojpeg=no' in sys.argv:
    sys.argv.remove('--turbojpeg=no')
    use_turbojpeg = False
else:
    use_turbojpeg = True

curfolder = os.path.dirname(os.path.abspath(__file__))
cpp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp')

#make sure the dependent third libs are all ready in 'thirdlibs/'
opencvincludedir = os.path.join(curfolder, 'thirdlibs/opencv/include')
opencvlibdir = os.path.join(curfolder, 'thirdlibs/opencv/lib')
pythonlibdir = os.path.join(curfolder, 'thirdlibs/python27/lib')
jpegturboincludedir = os.path.join(curfolder, 'thirdlibs/libjpeg-turbo/include')
jpegturbolibdir = os.path.join(curfolder, 'thirdlibs/libjpeg-turbo/lib')
    
opencvlibs = [
    'opencv_world', "IlmImf", "ippicv", "ippiw", "ittnotify", "libjasper",
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

pysource = 'python'
#build package of datareader
setup(name=modname,
    version=version,
    description="a package for data loading and preprocessing in training model",
    packages=find_packages(where=pysource),
    package_dir={'': pysource},
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions)
