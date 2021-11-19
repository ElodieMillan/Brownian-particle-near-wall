"""
Élodie Millan
Novembre 2021
Fonction calcul cumulant ordre 4 théorique.
"""

import numpy as np

def J(zth, Dpara, V, beta,):
    Ji = np.zeros(len(zth))
    print(zth)
    for i in range(len(zth)-1):
        print(i)
        Z = np.trapz(np.exp(beta * V(zth[:i])), zth[:i])
        P_eq_z = np.exp(beta * V(zth[:i])) / Z
        Dpara_mean = np.trapz(Dpara * P_eq_z, zth[:i])
        Ji[i] = np.trapz(np.exp(-beta*V(zth[:i])) * (Dpara(zth[:i])-Dpara_mean), zth[:i])

    return Ji

def R(z_th, Dpara, Dperp, V, beta,):
    Ri = np.zeros(len(z_th))
    for i in range(len(z_th)):
        Ri[i] = np.trapz( (J(z_th[:i], Dpara, V, beta) * np.exp(beta*V(zth[:i])) / Dperp(zth[:i]), zth[:i]) )

    return Ri

def C4_long(Dpara, Dperp, V, kBT, zth):
    """
    Le cumulant d'ordre 4 au temps long s'écrit:
    C4_long = 24*(D4*tau - C4)

    :param Dpara: Fonction coeficient de diffusion parallèle au mur.
    :param Dperp: Fonction coeficient de diffusion perpendiculaire au mur.
    :param potentiel: Potentiel subit par la particule.

    :return: C4, D4
    """

    global beta
    beta = 1/kBT
    Dpara_mean = np.trapz(Dpara * P_eq_z, zth[:i])
    #Calcul de D4
    Z = np.trapz(np.exp(beta*V(zth)), zth)
    P_eq_z = np.exp(beta*V(zth)) / Z
    D4 = np.trapz(( J(zth, Dpara, V, beta)**2 * exp(beta*V(zth))) / Dperp(zth), zth)
    J = [np.trapz(np.exp(-beta*V(zth[:i])) * (Dpara(zth[:i])-Dpara_mean), zth[:i]) for i in range(len(zth)-1)]
    # #calcul de C4
    # R = R(zth, Dpara, Dperp, V, beta)
    # R_mean = np.trapz(R*P_eq_z, zth)
    # f = np.exp(-beta*V(zth)) * (R_mean - R) / Z
    # C4 = np.trapz(f**2 / P_eq_z, zth)

    return D4 #, C4

