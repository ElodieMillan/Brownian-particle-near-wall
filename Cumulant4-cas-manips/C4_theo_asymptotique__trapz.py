"""
Élodie Millan
Février 2022
Fonction calcul cumulant ordre 4 théorique asymptotique.
"""

import numpy as np
# from scipy.integrate import simpson

def C4_court(D, Peq, kBT, B, lD, lB, H, a, eta, dx,):
    """
    Le cumulant d'ordre 4 au temps court s'écrit:
    C4_court = A4 * tau²
             = (<Dpara²> - <Dpara>²)/2 * tau².

    :param Dpara: Fonction coeficient de diffusion parallèle au mur.
    :param V: Fonction du potentiel subit par la particule.
    :param kBT: Valeur de l'energie thermique kB*T.
    :param hmin: Borne inférieur d'intégration.
    :param hmax: Borne supérieur d'intégration.
    :param dx: pas d'integration numérique.

    :return: A4
    """
    global beta
    beta = 1/kBT
    hmin = -H + H * 1e-5
    hmax = +H - H * 1e-5

    Nt = int((hmax-hmin)/dx)
    z = np.linspace(hmin, hmax, Nt, endpoint=True)

    N = np.trapz(Peq(z, B, lD, lB, H), z)

    Mean_Dpara = np.trapz(Peq(z, B, lD, lB, H)/N *D(z, a, eta, H), z)
    Mean_Dpara2 = np.trapz(Peq(z, B, lD, lB, H)/N *D(z, a, eta, H)**2, z)

    A4 = (Mean_Dpara2 - Mean_Dpara**2)*12

    return A4


def C4_long(Dpara, Dperp, Peq, H, dx):
    """
    Le cumulant d'ordre 4 au temps long s'écrit:
    C4_long = 24*(D4*tau - C4)

    :param Dpara: Fonction coeficient de diffusion parallèle au mur.
    :param Dperp: Fonction coeficient de diffusion perpendiculaire au mur.
    :param V: Fonction du potentiel subit par la particule.
    :param kBT: Valeur de l'energie thermique kB*T.
    :param a: Borne inférieur d'intégration.
    :param b: Borne supérieur d'intégration.
    :param dx: pas d'integration numérique.

    :return: 24*D4, 24*C4
    """

    espilon = H*1e-5

    hmin = -(H-espilon)
    hmax = H-espilon

    Nt = int((hmax - hmin) / dx)
    z = np.linspace(hmin, hmax, Nt) #, endpoint=True

    N = np.trapz(Peq(z), z)

    Dpara_mean = np.trapz(Dpara(z)*Peq(z) / N, z)

    def J(z):
        if z == hmin:
            return 0
        zp = np.linspace(hmin, z, int((z - hmin)/ dx))
        return np.trapz(Peq(zp)* (Dpara(zp)-Dpara_mean), zp)

    JJ = [J(i)**2 for i in z]
    D4 = np.trapz(JJ / Peq(z) / Dperp(z) / N, z)

    # def R(z):
    #     if z == hmin:
    #         return 0
    #     zp = np.linspace(hmin, z, int((z-hmin)/ dx), endpoint=True)
    #     r = np.trapz(y=[J(i)*np.exp(beta*V(i)) / Dperp(i) for i in zp], dx=dx)
    #     return r
    # 
    # RR = [R(i) for i in z]
    # R_mean = np.trapz(y=[RR[i] * Peq[i] for i in range(len(z))], dx=dx)
    # R_mean2 = np.trapz (y=[RR[i]**2 * Peq[i] for i in range(len(z))], dx=dx)
    # C4 = R_mean2 - R_mean**2

    return D4*24#, C4*24
