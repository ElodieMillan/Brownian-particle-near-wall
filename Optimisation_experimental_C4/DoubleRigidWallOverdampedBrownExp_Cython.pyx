#cython: language_level=3

"""
Élodie Millan
Janvier 2022
Particule confinée entre deux mur à z=+/- Hp.
Le centre de la particule ne peux pas aller plus loin que +H = Hp-a et -H = -(Hp-a).
-> comme le set up expérimentale de Maxime/Yacine/Nicolas.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, log, cosh, sinh
from libc.stdlib cimport rand, RAND_MAX
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from scipy import interpolate


sys.path.append(r"../OptimizedCython")
from InertialLangevin3D_cython import InertialLangevin3D


class RigidWallOverdampedLangevin3D( InertialLangevin3D ):
    def __init__(self, dt, Nt, a, H, lD, lB, B=4.8, Nt_sub=1, rho=1050.0, rhoF=1000.0,  eta=0.001, kBT=1.38e-23*300, x0=None):

        delta_z = a*0.1
        super().__init__(dt, Nt, a, x0=x0)
        self.a = a
        self.H = H
        self.Nt_sub = Nt_sub
        self.rho = rho
        self.rhoF = rhoF
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

    def MSD(self, axis, space=None, plot=True, output=False):
        """

        :param space: Choose between "bulk" and "wall".
        :param plot: Plot MSD is True.
        :param output: Return {tau, MSD(tau)} is True.
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


        if plot:
            plt.loglog(self.dt*list_dt_MSD, msd, "o", label="Numerical")

            plt.title(r"Mean square displacement on $" + axis +"$")
            plt.xlabel(r"$\tau$ $(\mathrm{s})$", fontsize=15)
            plt.ylabel(r"$\langle ($$" + axis + r"$$(t+\tau) - $$" + axis + r"$$(t))^2 \rangle$ $(\mathrm{m}^2)$", fontsize=15)

            ax = plt.gca()
            locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
            ax.xaxis.set_major_locator(locmaj)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
            ax.yaxis.set_major_locator(locmaj)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

            plt.legend()
            plt.show()

        if output:
            return msd, self.dt*self.Nt_sub * list_dt_MSD


    def Cumulant4(self, axis, plot=True, output=False):
        """

        :param axis: choose between "x", "y" or "z".
        :param plot: Plot show if True.
        :param output: Return {tau, cumulant4} if True.
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

        list_dt_c4 = np.array([], dtype=int)
        for i in range(len(str(self.Nt)) - 3):
            # Take just 10 points by decade.
            list_dt_c4 = np.concatenate(
                (
                    list_dt_c4,
                    np.arange(10 ** i, 10 ** (i + 1), 10 ** i, dtype=int),
                )
            )
        c4 = np.zeros(len(list_dt_c4))

        # --- Compute cumulant4
        for k, i in enumerate(list_dt_c4):
            if i == 0:
                c4[k] = 0
                continue
            deltaX = position[i:] - position[:-i]
            c4[k] = (np.mean(deltaX**4) - 3 * (np.mean(deltaX**2))**2) / (24)

        if plot:
            plt.loglog(self.dt*list_dt_c4, c4, "o", label=r"$\mathrm{Simulation}$")

            plt.xlabel(r"$\tau~(\mathrm{s})$", fontsize=15)
            plt.ylabel(r"$C^{(4)_"+axis+"}~(\mathrm{m}^4)$", fontsize=15)
            plt.axis([np.min(self.dt*list_dt_c4) / 5, np.max(self.dt*list_dt_c4) * 5, None, None])

            ax = plt.gca()
            locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
            ax.xaxis.set_major_locator(locmaj)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
            ax.yaxis.set_major_locator(locmaj)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            ax.yaxis.set_minor_locator(locmin)
            # ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

            plt.legend()
            # plt.show()

        if output:
            return self.dt*self.Nt_sub * list_dt_c4, c4



    def PDF(self, axis, N_tau=10, space=None, bins=50, plot=True, output=False):
        """

        :param axis: choose between "x", "y", "z", "dx", "dy" or "dz".
        :param N_tau: For displacements "dx", "dy" or "dz", choose lag time tau = N_tau * dt. (Default = 10).
        :param space: Choose if "bulk" or "wall".
        :param bins: Number of bins in histogramme. (Default = 50).
        :param plot: Plot show if True.
        :param output: Return {BinsPositions, Histogramme} is True.
        """
        B = 4.8
        tau_c = self.lB*self.a / self.D #equation (5.2.11) thèse Maxime :)
        tau = N_tau * self.dt

        # ------ What do you want ? ----
        if axis == "x":
            Axis = axis
            position = self.x

        elif axis == "y":
            Axis = axis
            position = self.y

        elif axis == "z":
            Axis = axis
            position = self.z

        elif axis == "dx":
            Axis = "\Delta x"
            position = self.x
            dX = position[N_tau:] - position[:-N_tau]
            std_num = np.std(dX)

        elif axis == "dy":
            Axis = "\Delta y"
            position = self.y
            dX = position[N_tau:] - position[:-N_tau]
            std_num = np.std(dX)

        elif axis == "dz":
            Axis = "\Delta z"
            position = self.z
            dX = position[N_tau:] - position[:-N_tau]
            std_num = np.std(dX)

        else:
            raise ValueError('WRONG AXIS : choose between positions "x", "y" and "z" or displacements "dx", "dy" and '
                             '"dz" !')

        # --------- Verification space
        if space != "bulk" and space != "wall":
            raise ValueError('WRONG SPACE : choose between "bulk" and "wall" !')


        # --------- PDF on the good space
        if space == "bulk":
            hist, bin_edges = np.histogram(dX, bins=bins, density=True)
            binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
            binsPositions = binsPositions / np.sqrt(2 * self.D * self.dt * N_tau)
            pdf = hist / np.trapz(hist, binsPositions)


        if space == "wall":
            if axis == "x" or axis == "y" or axis == "z":
                hist, bin_edges = np.histogram(position[position < 3e-6], bins=bins, density=False)
                binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
                pdf = hist / np.trapz(hist, binsPositions)

            else:
                hist, bin_edges = np.histogram(dX, bins=bins, density=False)
                binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
                pdf = hist / np.trapz(hist, binsPositions)
                binsPositions = binsPositions/std_num


        if plot:
            if space == "bulk":
                plt.semilogy(binsPositions, pdf, "o", markersize=4, label=r"$\mathrm{Numerical}$")


            if space == "wall":
                plt.semilogy(binsPositions, pdf, "o", label=r"$\mathrm{Numerical}$")


            plt.xlabel(r"$" + Axis + "/ \sqrt{2D \Delta t} $")
            plt.ylabel(r"$P(" + Axis + ") $")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.legend()
            plt.show()

        if output:
            return binsPositions, pdf

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



"""
CYTHON We put all methode as function out of the class (Object as Cython is complex).
"""
cdef double pi = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double gamma_xy_eff(double zi_1, double a, double eta, double H):
    """
    Formule de Libshaber
    """
    # Mur Top
    cdef double gam_xy_T = (
        6
        * pi
        * a
        * eta
        * (
            1
            - ((9 * a) / (16 * ((H-zi_1) + a)))
            + (a / (8 * ((H-zi_1) + a))) ** 3
            - (45 * a / (256 * ((H-zi_1) + a))) ** 4
            - (a / (16 * ((H-zi_1) + a))) ** 5
        )
        ** (-1)
    )

    cdef double gam_xy_B = (
        6
        * pi
        * a
        * eta
        * (
            1
            - ((9 * a) / (16 * ((H+zi_1) + a)))
            + (a / (8 * ((H+zi_1) + a))) ** 3
            - (45 * a / (256 * ((H+zi_1) + a))) ** 4
            - (a / (16 * ((H+zi_1) + a))) ** 5
        )
        ** (-1)
    )

    cdef double gam_xy_0 = 6 * pi * a * eta

    return (gam_xy_T + gam_xy_B - gam_xy_0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double gamma_z_eff(double zi_1, double a, double eta, double H):
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


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef double Dprime(double zi, double kBT, double eta, double a, double H ):
#     """
#     :return: La dérivée du coef de diffusion D'(z).
#     """
#
#     cdef double gammaT_prime = 6*pi*eta*a * (+42*a*(H-zi)**2 + 24*a**2*(H-zi) + 4*a**2) / (6*(H-zi)**2 + 2*a*(H-zi))**2
#     cdef double gammaB_prime = 6*pi*eta*a * (-42*a*(H+zi)**2 - 24*a**2*(H+zi) - 4*a**2) / (6*(H+zi)**2 + 2*a*(H+zi))**2
#
#     cdef double D_prime = - kBT*(gammaB_prime + gammaT_prime) / (gamma_z_eff(zi, a, eta, H))**2
#
#     return D_prime

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

    return  - kBT / (6*np.pi*a) * eta_eff_primes / eta_eff**2



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

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void assign_random_gaussian_pair(double[:] out, int assign_ix):
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * w

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def my_gaussian_fast(int n):
    cdef int i
    cdef double[:] result = np.zeros(n, dtype='f8', order='C')
    for i in range(n // 2):  # Int division ensures trailing index if n is odd.
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[:,:] trajectory_cython(unsigned long int Nt,
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








def test():
    langevin3D = RigidWallOverdampedLangevin3D(
        dt=1 / 60, Nt=1000000, a=1.5e-6, rho=1050, x0=(0.0, 0.0, 1.0e-6)
    )
    langevin3D.trajectory()
if __name__ == '__main__':
    test()
