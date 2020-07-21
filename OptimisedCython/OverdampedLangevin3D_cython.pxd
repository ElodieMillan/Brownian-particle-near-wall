# Élodie millan
# July 2020
# Equivalent to a ".h" for "OverdampedLangevin3D_cython.pyx" (cython structur)

import numpy as np
cimport numpy as np
from libcpp cimport bool


cdef class Langevin3D:
    cdef public float dt
    cdef public int Nt
    cdef public float R
    cdef public float eta
    cdef public float T
    cdef public (float, float, float) x0
    cdef public float kb
    cdef public float gamma
    cdef public float a
    cdef public float D
    cdef public np.ndarray t
    cdef public np.ndarray x
    cdef public np.ndarray y
    cdef public np.ndarray z

    cdef void trajectory(self)

    cdef void plotTrajectory(self)

    cdef void MSD1D(self, char axis, bool output, bool plot)

    cdef void MSD3D(self, output, plot)

    cdef void speedDistribution1D(self, axis, nbTimesIntervalle, bins, output, plot)

    cdef void dXDistribution1D(self, axis, nbTimesIntervalle, bins, output, plot)
