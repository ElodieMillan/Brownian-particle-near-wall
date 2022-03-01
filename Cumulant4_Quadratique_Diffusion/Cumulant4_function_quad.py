"""
Élodie Millan
Novembre 2021
Fonction calcul cumulant ordre 4 théorique.
"""

import numpy as np
from scipy.integrate import quad

def C4_long(Dpara, Dperp, V, kBT, a, b):
    """
    Le cumulant d'ordre 4 au temps long s'écrit: C4_long = 24*(D4*tau - C4)

    :param Dpara: Fonction coeficient de diffusion parallèle au mur.
    :param Dperp: Fonction coeficient de diffusion perpendiculaire au mur.
    :param V: Fonction du potentiel subit par la particule.
    :param kBT: Valeur de l'energie thermique kB*T.
    :param a: Borne d'intégration basse.
    :param b: Borne d'intégration haute.

    :return: C4, D4
    """

    global beta
    beta = 1 / kBT

    """ CALCUL DE D4
    """
    Z = quad(lambda zp: np.exp(beta * V(zp)), a, b)[0]
    P_eq_z = lambda z: np.exp(beta * V(z)) / Z

    Dpara_mean = quad(lambda zp: Dpara(zp) * P_eq_z(zp), a, b)[0]

    # Jfun = lambda zp : np.exp(-beta*V(zp)) * (Dpara(zp)-Dpara_mean)

    J = lambda z: (
        quad(lambda zp: np.exp(-beta * V(zp)) * (Dpara(zp) - Dpara_mean), a, z)[0]
    )

    D4 = (
        quad(
            lambda zp: (J(zp) ** 2 * np.exp(beta * V(zp))) / Dperp(zp), a, b
        )[0]
        / Z
    )

    # ----- calcul de C4
    R = lambda z: quad(lambda zp: J(zp) * np.exp(beta * V(zp)) / Dperp(zp), a, z)[
        0
    ]

    R_mean = quad(lambda zp: R(zp) * P_eq_z(zp), a, b)[0]

    f = lambda z: np.exp(-beta * V(z)) * (R_mean - R(z)) / Z

    C4 = quad(lambda zp: f(zp) ** 2 / P_eq_z(zp), a, b)[0]

    return D4, C4
