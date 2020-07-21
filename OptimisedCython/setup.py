from distutils.core import setup
from Cython.Build import cythonize
import numpy

# ---------------------------------------
# ------- Overdamped without wall -------
# ---------------------------------------

# setup(
#     ext_modules=cythonize("OverdampedLangevin3D_cython.pyx"),
#     include_dirs=[numpy.get_include()],
# )

# ---------------------------------------
# ------- Inertial without wall ---------
# ---------------------------------------

setup(
    ext_modules=cythonize("InertialLangevin3D_cython.pyx"),
    include_dirs=[numpy.get_include()],
)
