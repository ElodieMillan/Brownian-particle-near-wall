# Élodie millan
# July 2020
# Equivalent to a ".h" for "OverdampedLangevin3D_cython.pyx" (cython structur)

import numpy as np
cimport numpy as np
from libcpp cimport bool


cdef class Langevin3D:
    cdef double dt 
    cdef unsigned long long int Nt
    cdef double R
    cdef double eta
    cdef double T
    cdef (double, double, double) x0
    
    cdef double kb
    cdef double gamma
    cdef double a
    cdef double D
    cdef np.ndarray t
    cdef np.ndarray x
    cdef np.ndarray y
    cdef np.ndarray z
    
    cdef void trajectory(self)
