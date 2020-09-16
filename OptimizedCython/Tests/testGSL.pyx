'''
Gibbs sampler for function:

f(x,y) = x x^2 \exp(-xy^2 - y^2 + 2y - 4x)

using conditional distributions:

x|y \sim Gamma(3, y^2 +4)
y|x \sim Normal(\frac{1}{1+x}, \frac{1}{2(1+x)})

Original version written by Flavio Coelho.
Tweaked by Chris Fonnesbeck.
Ported to CythonGSL Thomas V. Wiecki.
'''
cimport cython
from cython_gsl cimport *

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from libc.math cimport sqrt

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double rng(double sigma):
    return gsl_ran_gaussian(r, sigma)


cdef test():
    cdef double sigma = 2
    cdef long int i
    cdef long int N = 1000000
    cdef double res
    for i in range(N):
        res = rng(sigma)

def a():
    test()
