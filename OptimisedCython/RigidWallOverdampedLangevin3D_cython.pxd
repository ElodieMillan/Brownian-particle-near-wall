# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from InertialLangevin3D_cython cimport InertialLangevin3D

cdef class RigidWallOverdampedLangevin3D(InertialLangevin3D):

    cdef np.float64_t rhoF
    cdef np.float64_t lD
    cdef np.float64_t g
    cdef np.float64_t delta_m
    cdef np.float64_t lB

    cdef np.float64_t _gamma_xy(self, np.float64_t zi_1)
    cdef np.float64_t _gamma_z(self, np.float64_t zi_1)
    cdef np.float64_t _a(self, np.float64_t gamma)
    cdef np.float64_t _PositionXi(self, np.float64_t xi_1, np.float64_t zi_1, np.float64_t rng)
    cdef np.float64_t _PositionZi(self, np.float64_t xi_1, np.float64_t zi_1, np.float64_t rng)
