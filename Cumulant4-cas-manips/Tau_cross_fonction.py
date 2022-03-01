"""
Élodie Millan
Novembre 2021
Fonction calcul cumulant ordre 4 théorique.
"""

import numpy as np
from scipy.integrate import quad

def C4_long(D_para, D_perp, V_theo, kBT, H, B, lD, a, eta):
    """
    Le cumulant d'ordre 4 au temps long s'écrit:
    C4_long = 24*(D4*tau - C4)

    :return: C4, D4
    """
    lB = kBT / (4 / 3 * np.pi * a ** 3 * (50) * 9.81)
    V = lambda z: V_theo(z, B, lD, lB, kBT)
    Dperp = lambda z: D_perp(z, lD, a, eta, kBT, H)
    Dpara = lambda z: D_para(z, lD, a, eta, kBT, H)

    global beta
    beta = 1 / kBT
    zz = np.linspace(-H, H, 1000)

    # Calcul de la pente D4
    Z = np.trapz(np.exp(-beta * V(zz)), zz)
    P_eq_z = np.exp(-beta * V(zz)) / Z

    Dpara_mean = np.trapz(Dpara(zz) * P_eq_z(zz), zz)


    J = lambda z: (quad(lambda zp: np.exp(-beta * V(zp)) * (Dpara(zp) - Dpara_mean), -H, z)[0])
    # J = lambda z: (np.trapz(np.exp(-beta * V(zz)) * (Dpara(zz) - Dpara_mean), zz))

    D4 = (quad(lambda zp: (J(zp) ** 2 * np.exp(beta * V(zp))) / Dperp(zp), a, b)[0]/ Z)
    D4 = np.trapz((J(zz)**2 * np.exp(beta * V(zz))) / Dperp(zz), zz)

    # # ----- calcul de C4
    # R = lambda z: quad(lambda zp: J(zp) * np.exp(beta * V(zp)) / Dperp(zp), -H, H)[0]
    #
    # R_mean = quad(lambda zp: R(zp) * P_eq_z(zp), -H, H)[0]
    #
    # f = lambda z: np.exp(-beta * V(z)) * (R_mean - R(z)) / Z
    #
    # C4 = quad(lambda zp: f(zp) ** 2 / P_eq_z(zp), -H, H)[0]

    return D4#, C4


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

    A4 = (Mean_Dpara2 - Mean_Dpara**2)/2

    return A4


def Cross_time(Dpara, Dperp, V, kBT, a, b):

    return (C4_long(Dpara, Dperp, V, kBT, a, b)[0] / C4_court(Dpara, V, kBT, a, b))