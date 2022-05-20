#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, log, cosh, sinh
from libc.stdlib cimport rand, RAND_MAX

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

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_gaussian():
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
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

    for i in range(1, Nt):
        for j in range(0, Nt_sub):
            x = positionXYi_cython(x, z, random_gaussian()/sqrt(dt), dt, a, eta, kBT, H)
            y = positionXYi_cython(y, z, random_gaussian()/sqrt(dt), dt, a, eta, kBT, H)
            z = positionZi_cython(z, random_gaussian()/sqrt(dt), dt, a, eta, kBT, H, lB, lD, B)

        res[0,i] = x
        res[1,i] = y
        res[2,i] = z

    return res