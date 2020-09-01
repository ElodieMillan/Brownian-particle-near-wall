# distutils: language = c++

# python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

# ---------------------------------------
# ------- Overdamped without wall -------
# ---------------------------------------

compiler_directives = {'language_level' : "3str",'boundscheck' : "False",'wraparound' : "False"}

# setup(
#     ext_modules=cythonize("OverdampedLangevin3D_cython.pyx", compiler_directives=compiler_directives,annotate=True),
#     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#     include_dirs=[numpy.get_include()],
# )

# # ---------------------------------------
# # ------- Inertial without wall ---------
# # ---------------------------------------

# setup(
#     ext_modules=cythonize("InertialLangevin3D_cython.pyx", compiler_directives=compiler_directives,annotate=True),
#     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#     include_dirs=[numpy.get_include()],
# )

setup(
    name="Langevin",
    ext_modules=cythonize("*.pyx",compiler_directives=compiler_directives),
    include_dirs=[numpy.get_include()],
)