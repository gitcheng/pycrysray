from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

exts = [Extension("pycrysray", 
                  ["pycrysray.pyx"],
                  )]
setup(
    name='pycrysray',
    version='__version__',
    discription='Raytracing tool for scintillation crystal bars',
    author='Chih-hsiang Cheng',
    email='ahsiang.c@gmail.com',
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules = exts,
)
