"""
Élodie Millan
Novembre 2021
Fonction calcul cumulant ordre 4 théorique.
"""

import numpy as np
from scipy.integrate import quad
# def quad(f, a, b):
#     res = romberg(f, a, b)
#     return (res, 0)

def C4_long(Dpara, Dperp, V, kBT, a, b, limit = 50):
    """
    Le cumulant d'ordre 4 au temps long s'écrit:
    C4_long = 24*(D4*tau - C4)

    :param Dpara: Fonction coeficient de diffusion parallèle au mur.
    :param Dperp: Fonction coeficient de diffusion perpendiculaire au mur.
    :param V: Fonction du potentiel subit par la particule.
    :param kBT: Valeur de l'energie thermique kB*T.
    :param a: Borne inférieur d'intégration.
    :param b: Borne supérieur d'intégration.

    :return: 24*C4, 24*D4
    """

    global beta
    beta = 1 / kBT

    # ------ Calcul de la pente D4
    Z = quad(lambda zp: np.exp(-beta * V(zp)), a, b)[0]
    P_eq_z = lambda z: np.exp(-beta * V(z)) / Z

    Dpara_mean = quad(lambda zp: Dpara(zp) * P_eq_z(zp), a, b)[0]


    J = lambda z: (
        quad(lambda zp: np.exp(-beta * V(zp)) * (Dpara(zp) - Dpara_mean), a, z, limit=limit)[0]
    )

    D4 = (
        quad(
            lambda zp: (J(zp)**2 * np.exp(beta * V(zp))) / Dperp(zp), a, b, limit=limit
        )[0]
        / Z
    )

    # ----- calcul de C4
    R = lambda z: quad(lambda zp: J(zp) * np.exp(beta * V(zp)) / Dperp(zp), a, z, limit=limit)[0]

    R_mean = quad(lambda zp: R(zp) * P_eq_z(zp), a, b, limit=limit)[0]
    R_mean2 = quad(lambda zp: R(zp)**2 * P_eq_z(zp), a, b, limit=limit)[0]

    C4 = R_mean2 - R_mean**2

    # f = lambda z: np.exp(-beta * V(z)) * (R_mean - R(z)) / Z
    # C4 = quad(lambda zp: f(zp) ** 2 / P_eq_z(zp), a, b, limit=limit)[0]

    return D4*24, C4*24


def C4_court(Dpara, V, kBT, a, b):
    """
    Le cumulant d'ordre 4 au temps court s'écrit:
    C4_court = A4 * tau²
             = (<Dpara²> - <Dpara>²)/2 * tau².

    :param Dpara: Fonction coeficient de diffusion parallèle au mur.
    :param V: Fonction du potentiel subit par la particule.
    :param kBT: Valeur de l'energie thermique kB*T.
    :param a: Borne inférieur d'intégration.
    :param b: Borne supérieur d'intégration.

    :return: A4
    """
    global beta
    beta = 1/kBT

    Z = quad(lambda zp: np.exp(-beta * V(zp)), a, b)[0]
    P_eq_z = lambda z: np.exp(-beta * V(z)) / Z

    Mean_Dpara = quad(lambda z : P_eq_z(z)*Dpara(z), a, b)[0]
    Mean_Dpara2 = quad(lambda z : P_eq_z(z)*Dpara(z)**2, a, b)[0]

    A4 = (Mean_Dpara2 - Mean_Dpara**2)*12

    return A4


def Cross_time(Dpara, Dperp, V, kBT, a, b):
    A4 = C4_court(Dpara, V, kBT, a, b)
    D4, C4 = C4_long(Dpara, Dperp, V, kBT, a, b)

    return D4/(2*A4)