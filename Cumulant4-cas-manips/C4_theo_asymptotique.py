"""
Élodie Millan
Février 2022
Fonction calcul cumulant ordre 4 théorique asymptotique.
"""

import numpy as np
# from scipy.integrate import simpson

def C4_court(Dpara, V, kBT, hmin, hmax, dx):
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

    Nt = int((hmax-hmin)/dx)
    z = np.linspace(hmin, hmax, Nt, endpoint=True)

    Z = np.trapz(y=[np.exp(-beta * V(i)) for i in z], dx=dx)
    P_eq_z = lambda z: np.exp(-beta * V(z)) / Z

    Mean_Dpara = np.trapz(y=[P_eq_z(i)*Dpara(i) for i in z], dx=dx)
    Mean_Dpara2 = np.trapz(y=[P_eq_z(i)*Dpara(i)**2 for i in z], dx=dx)

    A4 = (Mean_Dpara2 - Mean_Dpara**2)*12

    return A4


def C4_long(Dpara, Dperp, V, kBT, H, dx):
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

    global beta
    beta = 1 / kBT
    espilon = 1e-4*H
    hmin = -(H+espilon)
    hmax = H+espilon

    Nt = int((hmax - hmin) / dx)
    z = np.linspace(hmin, hmax, Nt, endpoint=True)

    VV = [V(i) for i in z]
    D_Para = [Dpara(i) for i in z]
    D_Perp = [Dperp(i) for i in z]

    Z = np.trapz(y=[np.exp(-beta*VV[i]) for i in range(len(z))], dx=dx)
    Peq = [np.exp(-beta * V(i)) / Z for i in z]

    Dpara_mean = np.trapz(y=[Peq[i]*D_Para[i] for i in range(len(z))], dx=dx)

    def J(z):
        if z == hmin:
            return 0
        zp = np.linspace(hmin, z, int((z - hmin)/ dx), endpoint=True)
        j = np.trapz( y=[np.exp(-beta * V(i)) * (Dpara(i) - Dpara_mean) for i in zp] , dx=dx )
        return j

    JJ = [J(i) for i in z]
    D4 = np.trapz( y=[JJ[i]**2 * np.exp(beta*VV[i]) / D_Perp[i] for i in range(len(z))], dx=dx )

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
