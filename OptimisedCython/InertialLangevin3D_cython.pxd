# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from OverdampedLangevin3D_cython import Langevin3D
from OverdampedLangevin3D_cython cimport Langevin3D

#DTYPE = np.float64 # C type equivalent at a DTYPE_t
#ctypedef np.float64_t DTYPE_t

cdef class InertialLangevin3D(Langevin3D):

    cdef np.float64_t rho
    cdef np.float64_t b
    cdef np.float64_t c
    cdef np.float64_t xi

    cdef np.float64_t _PositionXi(self, double xi1, double xi2, double rng)