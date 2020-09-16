# distutils: language = c

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
from Cython.Distutils import build_ext
import cython_gsl

import glob

sources = glob.glob("*.pyx")



extensions = []

for i in sources:
    extension = Extension(i[:-4],
                          sources= [i],
                          language='c',
                          libraries=cython_gsl.get_libraries(),
                          library_dirs=[cython_gsl.get_library_dir()],
                          include_dirs=[cython_gsl.get_cython_include_dir(),numpy.get_include()],

                          )

    extensions.append(extension)

setup(
    name="Langevin",
    ext_modules = extensions,
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
)

