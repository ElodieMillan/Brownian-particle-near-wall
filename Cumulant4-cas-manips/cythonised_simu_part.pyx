#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, log, cosh, sinh
from libc.stdlib cimport rand, RAND_MAX, srand
import time


cdef double pi = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double gamma_xy_eff(double zi_1, double a, double eta, double H):
    """
    Formule de Libshaber
    """
    # Mur Top
    cdef double xi_T = a / ((H-zi_1) + a)
    cdef double gam_xy_T = (
        6.
        * pi
        * a
        * eta
        *
        (
            1.
            - 9./16. * xi_T
            + 1./8. * xi_T**3.
            - 45./256. * xi_T**4.
            - 1./16. * xi_T** 5.
        )
        ** (-1)
    )

    # Mur Bottom
    cdef double xi_B = a / ((H+zi_1) + a)
    cdef double gam_xy_B = (
        6
        * pi
        * a
        * eta
        * (
            1.
            - 9./16. * xi_B
            + 1./8. * xi_B**3
            - 45./256. * xi_B**4
            - 1./16. * xi_B** 5
        )
        ** (-1)
    )

    cdef double gam_xy_0 = 6 * pi * a * eta

    return (gam_xy_T + gam_xy_B - gam_xy_0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double gamma_z_eff(double zi_1, double a, double eta, double H):
    """
    Formule de Padé
    """
    # Mur Top
    cdef double gam_z = (
        6
        * pi
        * a
        * eta
        * (
            (
                (6 * (H-zi_1)**2 + 9*a*(H-zi_1) + 2*a**2)
                / (6 * (H-zi_1)**2 + 2*a*(H-zi_1))
            )
        )
    )
    # Mur Bottom
    cdef double gam_z_2 = (
        6
        * pi
        * a
        * eta
        * ((6*(H+zi_1)**2 + 9*a*(H+zi_1) + 2*a**2)/ (6 * (H+zi_1)**2 + 2*a*(H+zi_1)))
    )

    cdef double gam_z_0 = 6 * pi * a * eta

    return (gam_z + gam_z_2 - gam_z_0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double w(double gamma, double kBT):
    """
    :return: Le bruit multiplicatif.
    """
    cdef double noise = sqrt(2 * kBT / gamma)
    return noise

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionXYi_cython(double xi_1, double zi_1, double rng, double dt, double a,
                               double eta, double kBT, double H):
    """
    :return: Position parallèle de la particule au temps suivant t+dt.
    """
    cdef double gamma = gamma_xy_eff(zi_1, a, eta, H)  #gamma effectif avec 2 murs
    cdef double xi = xi_1 + w(gamma, kBT) * rng * dt

    return xi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double Dprime_z(double zi, double kBT, double eta, double a, double H):
    # Spurious force pour corriger overdamping (Auteur: Dr. Maxime Lavaud)

    cdef double eta_B_primes = -(a * eta * (2 * a ** 2 + 12 * a * (H + zi) + 21 * (H + zi) ** 2)) / (
        2 * (H + zi) ** 2 * (a + 3 * (H + zi)) ** 2
    )

    cdef double eta_T_primes = (
        a
        * eta
        * (2 * a ** 2 + 12 * a * (H-zi) + 21 * (H-zi) ** 2)
        / (2 * (a + 3*H - 3*zi) ** 2*(H-zi) ** 2)
    )

    cdef double eta_eff_primes = eta_B_primes + eta_T_primes

    cdef double eta_B = eta * (6*(H+zi)**2 + 9*a*(H+zi) + 2*a**2) / (6*(H+zi)**2 + 2*a*(H+zi))
    cdef double eta_T = eta * (6*(H-zi)**2 + 9*a*(H-zi) + 2*a**2) / (6*(H-zi)**2 + 2*a*(H-zi))

    cdef double eta_eff = eta_B + eta_T - eta

    return  - kBT / (6*pi*a) * eta_eff_primes / eta_eff**2



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double Forces(double z, double H, double kBT, double B, double lD, double lB):

    cdef double Felec = B * kBT/lD * exp(-H/lD) * (exp(-z/lD) - exp(z/lD))
    cdef double Fgrav = -kBT/lB
    return Felec + Fgrav


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionZi_cython(double zi_1, double rng, double dt, double a,
                               double eta, double kBT, double H, double lB, double lD, double B):
    """
    :return: Position perpendiculaire de la particule au temps suivant t+dt.
    """
    cdef double gamma = gamma_z_eff(zi_1, a, eta, H) #gamma effectif avec 2 murs

    cdef double zi = zi_1  + Dprime_z(zi_1, kBT, eta, a, H )*dt + Forces(zi_1, H, kBT, B, lD, lB)*dt /gamma +  w(gamma, kBT)*rng*dt
    if zi < -(H):
        zi = -2*H - zi
    if zi > H:
        zi =  2*H - zi

    return zi

## Random uniform classic
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX

## MERSENNE TWISTER ALGORITHM to generate a random number
cdef unsigned NN = 312
cdef unsigned MM = 156
cdef unsigned long long MATRIX_A = 0xB5026F5AA96619E9ULL
cdef unsigned long long UM = 0xFFFFFFFF80000000ULL
cdef unsigned long long LM = 0x7FFFFFFFULL
cdef unsigned long long mt[312]
cdef unsigned mti = NN + 1
cdef unsigned long long mag01[2]

@cython.nonecheck(False)
cdef mt_seed(unsigned long long seed):
    global mt
    global mti
    global mag01
    global NN
    global MATRIX_A
    mt[0] = seed
    for mti in range(1,NN):
        mt[mti] = (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti)
    mag01[0] = 0ULL
    mag01[1] = MATRIX_A
    mti = NN

@cython.nonecheck(False)
cdef unsigned long long genrand64():
    cdef int i
    cdef unsigned long long x
    global mag01
    global mti
    global mt
    global NN
    global MM
    global UM
    global LM
    if mti >= NN:
        for i in range(NN-MM):
            x = (mt[i]&UM) | (mt[i+1]&LM)
            mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[int(x&1ULL)]
        for i in range(NN-MM, NN-1):
            x = (mt[i]&UM)|(mt[i+1]&LM)
            mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[int(x&1ULL)]
        x = (mt[NN-1]&UM)|(mt[0]&LM)
        mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[int(x&1ULL)]
        mti = 0
    x = mt[mti]
    mti += 1
    x ^= (x >> 29) & 0x5555555555555555ULL
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL
    x ^= (x << 37) & 0xFFF7EEE000000000ULL
    x ^= (x >> 43);
    return x

# Seed the random number generator
@cython.nonecheck(False)
cdef seed_random(unsigned long long seed):
    """
    Seed the C random number generator with the current system time.
    :return: none
    """
    if seed == 0:
        mt_seed(time.time())
    else:
        mt_seed(seed)

#BOX-MULLER ALGORITHM : méthode polaire pour éviter les calculs de cos et sin
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double uniform_rv():
    """
    Generate a uniform random variable in [0,1]
    :return: (double) a random uniform number in [0,1]
    """
    return (genrand64() >> 11) * (1.0/9007199254740991.0)


@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_gaussian():
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * uniform_rv() - 1.0
        x2 = 2.0 * uniform_rv() - 1.0
        w = x1 * x1 + x2 * x2
    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double[:,:] trajectory_cython(unsigned long int Nt,
                                   unsigned long int Nt_sub,
                                   double[:,:] res,
                                   double dt,
                                   double a, double eta,
                                   double kBT, double H, double lB, double lD, double B):
    """    
    :return: Trajectoire X, Y, Z calculer avec Cython.
    """
    cdef unsigned long int i
    cdef unsigned long int j
    cdef double x = res[0,0]
    cdef double y = res[1,0]
    cdef double z = res[2,0]
    seed_random(0)

    for i in range(1, Nt):
        for j in range(0, Nt_sub):
            x = positionXYi_cython(x, z, random_gaussian()/sqrt(dt), dt, a, eta, kBT, H)
            y = positionXYi_cython(y, z, random_gaussian()/sqrt(dt), dt, a, eta, kBT, H)
            z = positionZi_cython(z, random_gaussian()/sqrt(dt), dt, a, eta, kBT, H, lB, lD, B)

        res[0,i] = x
        res[1,i] = y
        res[2,i] = z

    return res