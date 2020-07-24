# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libcpp cimport bool
DTYPE = np.float64 # C type equivalent at a DTYPE_t
ctypedef np.float64_t DTYPE_t

cdef class Langevin3D:
    cdef DTYPE_t dt 
    cdef unsigned long long int Nt
    cdef DTYPE_t R
    cdef DTYPE_t eta
    cdef DTYPE_t T
    cdef (DTYPE_t, DTYPE_t, DTYPE_t) x0

    cdef DTYPE_t tau
    cdef DTYPE_t m
    cdef DTYPE_t kb
    cdef DTYPE_t gamma
    cdef DTYPE_t a
    cdef DTYPE_t D
    cdef np.ndarray t
    cdef np.ndarray x
    cdef np.ndarray y
    cdef np.ndarray z
    cdef np.ndarray list_dt_MSD
    cdef np.ndarray MSD
    
    cdef void trajectory(self)
