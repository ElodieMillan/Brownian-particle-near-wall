"""
Élodie Millan
Janvier 2022
Particule confinée entre deux mur à z=+/- Hp.
Le centre de la particule ne peux pas aller plus loin que +H = Hp-a et -H = -(Hp-a).
-> comme le set up expérimentale de Maxime/Yacine/Nicolas.
"""

import numpy as np
from tqdm import tqdm
import sys
from scipy import interpolate
from cythonised_simu_part import trajectory_cython

sys.path.append(r"/home/e.millan/Documents/Stage2020-Nageurs-actifs-proche-de-parois-deformable/Code-Cython")

from OverdampedLangevin3D_cython import Langevin3D


class RigidWallOverdampedLangevin3D( Langevin3D ):
    def __init__(self, dt, Nt, a, H, lD, lB, B=4.8, Nt_sub=1, eta=0.001, kBT=1.38e-23*300, x0=None):

        delta_z = a*0.1
        super().__init__(dt, Nt, a, x0=x0)
        self.a = a
        self.H = H
        self.Nt_sub = Nt_sub
        self.eta = eta
        self.kBT = kBT
        self.Hp = self.H + self.a
        self.D0 = kBT/(6*np.pi*self.eta*self.a)
        self.lD = lD
        self.B = B
        self.lB = lB
        self.sample_f = self.sample()
        self.x0 = x0
        if x0 == None:
            self.x0 = (0.0, 0.0, self.return_samples(1))


        del self.t

    def trajectory(self, output=False):
        """
        :param output: Boolean - if true, return x, y, z (default : false).

        :return: return the x, y, z trajectory.
        """

        res = np.zeros((3, self.Nt))

        # First values of trajectory compute with initial value.
        res[0,0] = self.x0[0]
        res[1,0] = self.x0[1]
        res[2,0] = self.x0[2]

        res = np.asarray(trajectory_cython(self.Nt, self.Nt_sub,
                                   res,
                                   self.dt,
                                   self.a, self.eta,
                                   self.kBT, self.H, self.lB, self.lD, self.B))

        self.x = res[0,:]
        self.y = res[1,:]
        self.z = res[2,:]

        if output:
            return res

    ## SOME ANALYSIS FUNCTIONS
    def MSD(self, axis):
        """

        :param axis: Choose between "x", "y" and "z".

        :return (time, MSD(axis))
        """
        if axis == "x":
            position = self.x
        elif axis == "y":
            position = self.y
        elif axis == "z":
            position = self.z
        else:
            raise ValueError('WRONG AXIS : choose between "x", "y" and "z" !')

        list_dt_MSD = np.array([], dtype=int)
        for i in range(len(str(self.Nt)) - 1):
            # Take just 10 points by decade.
            list_dt_MSD = np.concatenate(
                (
                    list_dt_MSD,
                    np.arange(10 ** i, 10 ** (i + 1), 10 ** i, dtype=int),
                )
            )
        # -----------------------
        NumberOfMSDPoint = len(list_dt_MSD)
        msd = np.zeros(NumberOfMSDPoint)
        for k, i in enumerate(tqdm(list_dt_MSD)):
            if i == 0:
                msd[k] = 0
                continue
            msd[k] = np.mean((position[i:]-position[:-i])**2)

        return self.dt*self.Nt_sub * list_dt_MSD, msd


    def Cumulant4(self, axis):
        """

        :param axis: choose between "x", "y" or "z".

        :return (time, C4)
        """
        # --- def some array
        if axis == "x":
            position = self.x
        elif axis == "y":
            position = self.y
        elif axis == "z":
            position = self.z
        else:
            raise ValueError('WRONG AXIS : choose between "x", "y" and "z" !')

        list_Nt_c4 = np.array([], dtype=int)
        for i in range(len(str(self.Nt)) - 1):
            # Take just 10 points by decade.
            list_Nt_c4 = np.concatenate(
                (
                    list_Nt_c4,
                    np.arange(10 ** i, 10 ** (i + 1), 10 ** i, dtype=int),
                )
            )
        c4 = np.zeros(len(list_Nt_c4))

        # --- Compute cumulant4
        for k, i in enumerate(list_Nt_c4):
            if i == 0:
                c4[k] = 0
                continue
            deltaX = position[i:] - position[:-i]
            c4[k] = (np.mean(deltaX**4) - 3 * (np.mean(deltaX**2))**2)

        return self.dt*self.Nt_sub * list_Nt_c4, c4

    # Functions du problème.
    def D_para(self, z):
        D = [self.kBT / gamma_xy_eff(i, self.a, self.eta, self.H) for i in z]
        return np.asarray(D)

    def D_perp(self, z):
        D = [self.kBT / gamma_z_eff(z, self.a, self.eta, self.H) for i in z]
        return np.asarray(D)

    def P_z_wall(self, z):
        """
        :param A: Normalisation.
        :param B: 4.8 experimentally.

        :return: P_eq(z) near a wall.
        """
        if type(z) != np.ndarray:
            if (z > self.H) or (z < -self.H):
                return 0
            return np.exp(-self.B*np.exp(-self.H/self.lD) * (np.exp(-z/self.lD) + np.exp(z/self.lD)) - z / self.lB)
        Pz = lambda z : np.exp(-self.B*np.exp(-self.H/self.lD) * (np.exp(-z/self.lD) + np.exp(z/self.lD)) - z / self.lB)
        P = np.array([Pz(zz) for zz in z])
        P[z < -self.H] = 0
        P[z > self.H] = 0

        return P


    def _P_deltaZ_longTime(self, dz, Hp):

        z = np.linspace(-Hp, Hp, 1000)
        dP = self.P_z_wall(z, Hp) * self.P_z_wall(z+dz, Hp)
        P = np.trapz(dP, z)

        return P

    def P_deltaZ_longTime(self, dz, Hp):
        Pdf = np.array([self._P_deltaZ_longTime(i,Hp) for i in dz])
        Pdf = Pdf/np.trapz(Pdf, dz)

        return Pdf

    ## Tirage aléatoire sur la PDF(z) pour z_0
    def f(self, z):
    # does not need to be normalized
        zz = np.zeros_like(z)

        for n,i in enumerate(z):
            if (i < -self.H) or (i > self.H):
                zz[n] = 0
            else:
                zz[n] = np.exp(-self.B*np.exp(-self.H/self.lD) * (np.exp(-i/self.lD) + np.exp(i/self.lD)) - i / self.lB)

        return zz

    def sample(self):
        x = np.linspace(-self.H, self.H, 1000)
        y = self.f(x)  # probability density function, pdf
        cdf_y = np.cumsum(y)  # cumulative distribution function, cdf
        cdf_y = cdf_y / cdf_y.max()  # takes care of normalizing cdf to 1.0
        inverse_cdf = interpolate.interp1d(cdf_y, x)  # this is a function
        return inverse_cdf

    def return_samples(self, N=1):
    # let's generate some samples according to the chosen pdf, f(x)
        try :
            uniform_samples = np.random.random(int(N))
            required_samples = self.sample_f(uniform_samples)
            return required_samples[0]
        except ValueError:
            self.return_samples(N)



"""
FIN CLASSE
"""











def test():
    langevin3D = RigidWallOverdampedLangevin3D(
        dt=1 / 60, Nt=1000000, a=1.5e-6, rho=1050, x0=(0.0, 0.0, 1.0e-6)
    )
    langevin3D.trajectory()

if __name__ == '__main__':
    test()
