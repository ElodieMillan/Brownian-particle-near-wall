import numpy as np

kBT = 1.38e-23*300

def _gamma_z_eff(zi_1, a, eta, H):
    """
    Formule de Padé
    """
    # Mur Top
    gam_z = (
            6
            * np.pi
            * a
            * eta
            * (
                (
                        (6 * (H - zi_1) ** 2 + 9 * a * (H - zi_1) + 2 * a ** 2)
                        / (6 * (H - zi_1) ** 2 + 2 * a * (H - zi_1))
                )
            )
    )
    # Mur Bottom
    gam_z_2 = (
            6
            * np.pi
            * a
            * eta
            * (
                (
                        (6 * (H + zi_1) ** 2 + 9 * a * (H + zi_1) + 2 * a ** 2)
                        / (6 * (H + zi_1) ** 2 + 2 * a * (H + zi_1))
                )
            )
    )

    gam_z_0 = 6 * np.pi * a * eta

    return (gam_z + gam_z_2 - gam_z_0)


def _gamma_xy_eff(zi_1, a, eta, H):
    """
    Formule de Libshaber
    """
    # Mur Top
    xi_T = a / ((H - zi_1) + a)
    gam_xy_T = (
            6
            * np.pi
            * a
            * eta
            * (
                    1
                    - 9 / 16 * xi_T
                    + 1 / 8 * xi_T ** 3
                    - 45 / 256 * xi_T ** 4
                    - 1 / 16 * xi_T ** 5
            )
            ** (-1)
    )

    # Mur Bottom
    xi_B = a / ((H + zi_1) + a)
    gam_xy_B = (
            6
            * np.pi
            * a
            * eta
            * (
                    1
                    - 9 / 16 * xi_B
                    + 1 / 8 * xi_B ** 3
                    - 45 / 256 * xi_B ** 4
                    - 1 / 16 * xi_B ** 5
            )
            ** (-1)
    )

    gam_xy_0 = 6 * np.pi * a * eta

    return (gam_xy_T + gam_xy_B - gam_xy_0)


def Dprime_z_eff(zi, kBT, eta, a, H):
    # Spurious force pour corriger overdamping (Auteur: Dr. Maxime Lavaud)
    eta_B = lambda zi: eta * (6 * (H + zi) ** 2 + 9 * a * (H + zi) + 2 * a ** 2) / (
                6 * (H + zi) ** 2 + 2 * a * (H + zi))
    eta_T = lambda zi: eta * (6 * (H - zi) ** 2 + 9 * a * (H - zi) + 2 * a ** 2) / (
                6 * (H - zi) ** 2 + 2 * a * (H - zi))

    eta_B_primes = -(a * eta * (2 * a ** 2 + 12 * a * (H + zi) + 21 * (H + zi) ** 2)) / (
            2 * (H + zi) ** 2 * (a + 3 * (H + zi)) ** 2
    )
    eta_T_primes = (
            a
            * eta
            * (2 * a ** 2 + 12 * a * (H - zi) + 21 * (H - zi) ** 2)
            / (2 * (a + 3 * H - 3 * zi) ** 2 * (H - zi) ** 2)
    )

    eta_eff = eta_B(zi) + eta_T(zi) - eta
    eta_eff_prime = eta_B_primes + eta_T_primes

    return - kBT / (6 * np.pi * a) * eta_eff_prime / eta_eff ** 2

def V(z, B, lD, lB):
    return B * kBT * np.exp(-H / lD) * (np.exp(-z / lD) + np.exp(+z / lD)) + kBT / lB * z

def D_perp(z, a, eta, H):
    return kBT / _gamma_z_eff(z, a, eta, H)

def D_para(z, a, eta, H):
    return kBT / _gamma_xy_eff(z, a, eta, H)

def D_0(z, a, eta):
    return kBT / (6 * np.pi * eta * a * np.ones(len(z)))

def F_elec(z, B, lD):
    return B * kBT / lD * np.exp(-H / lD) * (np.exp(-z / lD) - np.exp(z / lD))

def F_grav(z, lB,):
    return - kBT / lB * np.ones(len(z))

def P_eq(z, B, lD, lB, H):
    return np.exp(-B * np.exp(-H / lD) * (np.exp(-z / lD) + np.exp(z / lD)) - z / lB)

def F_spurious(z, a, eta, H):
    return Dprime_z_eff(z, kBT, eta, a, H) * gamma_z_eff(z, eta, a, H)

def gauss(x, mu, std):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-((x - mu) / std) ** 2 / 2)


def P_D(len_dZ, Diff, a, eta, B, lD, lB, H):
    # Computing the D PDF.
    epsilon=H*1e-8
    z = np.linspace(-(H-epsilon), (H-epsilon), len_dZ)
    P_D = Diff(z, a, eta, H) * P_eq(z, B, lD, lB, H)
    P_D = P_D / np.trapz(P_D, z) # extra step to ensure PDF normalization
    return Diff(z, a, eta, H), P_D

def _P_Di_short_time(dZ, tau, Diff, len_dZ, a, eta,  B, lD, lB, H):
    # Using the D PDF to compute P()
    D_z, P_Di = P_D(len_dZ, Diff, a, eta, B, lD, lB, H)
    P = P_Di / np.sqrt(4 * np.pi * D_z * tau) * np.exp(-(dZ ** 2) / (4 * D_z * tau))
    P = np.trapz(P, D_z)
    return P

# Creating a handy function for easier use with Dz numpy arrays
def P_Di_short_time(dZ, tau, Diff, a, eta, B, lD, lB, H):
    P = np.array([_P_Di_short_time(i, tau, Diff, len(dZ), a, eta, B, lD, lB, H) for i in dZ])
    P = P / np.trapz(P, dZ)  # extra step to ensure PDF normalization
    return P


def _Pdeltaz_long(dZ, B, lD, lB, H):
    z = np.linspace(-H, +H, 1000)
    dP = P_eq(z, B, lD, lB, H) * P_eq(z + dZ, B, lD, lB, H)
    P = np.trapz(dP,z)
    return P

def Pdeltaz_long(dZ, B, lD, lB, H):
    pdf = np.array([_Pdeltaz_long(i, B, lD, lB, H) for i in dZ])
    pdf = pdf / np.trapz(pdf,dZ)
    return pdf
