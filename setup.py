from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

exts = [Extension("pycrysray", 
                  ["pycrysray.pyx"],
                  )]
setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    ext_modules = exts,
)
