from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

exts = [Extension("pycrysray", 
                  ["pycrysray.pyx"],
              ),
        Extension("crysraytests",
                  ["crysraytests.pyx"],
              ),]
setup(
    name='pycrysray',
    version='1.1',
    discription='Raytracing tool for scintillation crystal bars',
    author='Chih-hsiang Cheng',
    email='ahsiang.c@gmail.com',
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules = cythonize(exts),
)
