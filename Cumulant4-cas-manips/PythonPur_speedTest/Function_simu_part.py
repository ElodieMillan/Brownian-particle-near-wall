import numpy as np

pi = np.pi

def gamma_xy_eff(zi_1, a, eta, H):
    """
    Formule de Libshaber
    """
    # Mur Top
    xi_T = a / ((H-zi_1) + a)
    gam_xy_T = (
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
    xi_B = a / ((H+zi_1) + a)
    gam_xy_B = (
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

    gam_xy_0 = 6 * pi * a * eta

    return (gam_xy_T + gam_xy_B - gam_xy_0)


def gamma_z_eff(zi_1, a, eta, H):
    """
    Formule de Padé
    """
    # Mur Top
    gam_z = (
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
    gam_z_2 = (
        6
        * pi
        * a
        * eta
        * ((6*(H+zi_1)**2 + 9*a*(H+zi_1) + 2*a**2)/ (6 * (H+zi_1)**2 + 2*a*(H+zi_1)))
    )

    gam_z_0 = 6 * pi * a * eta

    return (gam_z + gam_z_2 - gam_z_0)


def w(gamma, kBT):
    """
    :return: Le bruit multiplicatif.
    """
    noise = np.sqrt(2 * kBT / gamma)
    return noise

def positionXYi_cython(xi_1, zi_1, rng, dt, a,
                               eta, kBT, H):
    """
    :return: Position parallèle de la particule au temps suivant t+dt.
    """
    gamma = gamma_xy_eff(zi_1, a, eta, H)  #gamma effectif avec 2 murs
    xi = xi_1 + w(gamma, kBT) * rng * dt

    return xi


def Dprime_z(zi, kBT, eta, a, H):
    # Spurious force pour corriger overdamping (Auteur: Dr. Maxime Lavaud)

    eta_B_primes = -(a * eta * (2 * a ** 2 + 12 * a * (H + zi) + 21 * (H + zi) ** 2)) / (
        2 * (H + zi) ** 2 * (a + 3 * (H + zi)) ** 2
    )

    eta_T_primes = (
        a
        * eta
        * (2 * a ** 2 + 12 * a * (H-zi) + 21 * (H-zi) ** 2)
        / (2 * (a + 3*H - 3*zi) ** 2*(H-zi) ** 2)
    )

    eta_eff_primes = eta_B_primes + eta_T_primes

    eta_B = eta * (6*(H+zi)**2 + 9*a*(H+zi) + 2*a**2) / (6*(H+zi)**2 + 2*a*(H+zi))
    eta_T = eta * (6*(H-zi)**2 + 9*a*(H-zi) + 2*a**2) / (6*(H-zi)**2 + 2*a*(H-zi))

    eta_eff = eta_B + eta_T - eta

    return  - kBT / (6*pi*a) * eta_eff_primes / eta_eff**2


def Forces(z, H, kBT, B, lD, lB):

    Felec = B * kBT/lD * np.exp(-H/lD) * (np.exp(-z/lD) - np.exp(z/lD))
    Fgrav = -kBT/lB
    return Felec + Fgrav


def positionZi_cython(zi_1, rng, dt, a,
                               eta, kBT, H, lB, lD, B):
    """
    :return: Position perpendiculaire de la particule au temps suivant t+dt.
    """
    gamma = gamma_z_eff(zi_1, a, eta, H) #gamma effectif avec 2 murs

    zi = zi_1  + Dprime_z(zi_1, kBT, eta, a, H )*dt + Forces(zi_1, H, kBT, B, lD, lB)*dt /gamma +  w(gamma, kBT)*rng*dt
    if zi < -(H):
        zi = -2*H - zi
    if zi > H:
        zi =  2*H - zi

    return zi

def random_gaussian(): #Box-muller algorithm : méthode polaire pour éviter les calculs de cos et sin
    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * np.random.uniform(low=0.0, high=1.0) - 1.0
        x2 = 2.0 * np.random.uniform(low=0.0, high=1.0) - 1.0
        w = x1 * x1 + x2 * x2
    w = ((-2.0 * np.log(w)) / w) ** 0.5
    return x1 * w

def trajectory_cython(Nt, Nt_sub, res, dt, a, eta, kBT, H, lB, lD, B):
    """    
    :return: Trajectoire X, Y, Z calculer avec Cython.
    """
    x = res[0,0]
    y = res[1,0]
    z = res[2,0]

    for i in range(1, Nt):
        for j in range(0, Nt_sub):
            x = positionXYi_cython(x, z, random_gaussian()/np.sqrt(dt), dt, a, eta, kBT, H)
            y = positionXYi_cython(y, z, random_gaussian()/np.sqrt(dt), dt, a, eta, kBT, H)
            z = positionZi_cython(z, random_gaussian()/np.sqrt(dt), dt, a, eta, kBT, H, lB, lD, B)

        res[0,i] = x
        res[1,i] = y
        res[2,i] = z

    return res