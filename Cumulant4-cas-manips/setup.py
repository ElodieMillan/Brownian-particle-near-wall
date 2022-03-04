#cython: language_level=3
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
                            #extra_compile_args=['/O2'],
                            language='c',
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])

    extensions.append(extension)

for e in extensions:
    e.cython_directives = {'language_level': "3str"} #all are Python-3


setup(
    name="Langevin",
    ext_modules = extensions,
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
)


