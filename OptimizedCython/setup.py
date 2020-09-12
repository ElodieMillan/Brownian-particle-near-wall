# distutils: language = c

# python setup.py build_ext -i clean

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
from Cython.Distutils import build_ext

import glob

sources = glob.glob("*.pyx")



compiler_directives = {'language_level' : "3str",'boundscheck' : "False",'wraparound' : "False"}

extensions = []

for i in sources:
    extension = Extension(i[:-4],
                            sources= [i],
                            extra_compile_args=['/O2'],
                            language='c',)

    extensions.append(extension)

setup(
    name="Langevin",
    ext_modules = extensions,
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
)


