#cython: language_level=3

"""
Élodie Millan
June 2020
Langevin equation 3D for a free particule close to a rigid wall without inertia and with weight.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport rand, RAND_MAX
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


sys.path.append(r"../OptimizedCython")
from InertialLangevin3D_cython import InertialLangevin3D


class RigidWallOverdampedLangevin3D( InertialLangevin3D ):
    def __init__(self, dt, Nt, a, H, Hs, lB, Nt_sub=1, rho=1050.0, rhoF=1000.0, kBT=1.38e-23*300, x0=None):
        """
        :param dt: float - Time step [s].
        :param Nt: int - Number of time points.
        :param a: float - Radius of particule [m].
        :param rho: float - Volumic mass of the particule [kg/m³] (DEFAULT = 1050 kg/m³).
        :param rhoF: float - Volumic mass of the fluid [kg/m³] (DEFAULT = 1000 kg/m³).
        :param eta: float - Fluid viscosity (DEFAULT = 0.001 Pa/s).
        :param T: float - Temperature (DEFAULT = 300 K).
        :param x0: array float - Initial position of particule (DEFAULT = (0,0,a) [m]).
        """
        if x0 == None:
            x0 = (0.0, 0.0, a)
        super().__init__(dt, Nt, a, x0=x0)
        self.a = a
        self.H = H
        self.Hs = Hs
        self.Nt_sub = Nt_sub
        self.rhoF = rhoF
        self.kBT = kBT
        self.Hp = self.H + self.a
        self.Db = kBT/(6*np.pi*0.001*a)
        self.Dperp_0 = (self.H - self.a)/(2*self.a) * self.Db
        self.Dpara_0 = (self.H - self.a)/(2*self.a) * self.Db
        del self.t
        self.lB = lB

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
                                   self.a,  self.Dpara_0, self.Dperp_0,
                                   self.kBT, self.H, self.Hs, self.lB))

        self.x = res[0,:]
        self.y = res[1,:]
        self.z = res[2,:]

        if output:
            return self.x, self.y, self.z

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
            return msd, self.dt*self.Nt_sub*list_dt_MSD


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
        for k, i in enumerate((list_dt_c4)):
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


    def P_z_wall(self, z, Hp):
        """

        :param A: Normalisation.
        :param B: 4.8 experimentally.

        :return: P_eq(z) near a wall.
        """
        if type(z) != np.ndarray:
            if (z > Hp) or (z < -Hp):
                return 0
            return 1/(2*(Hp))

        P = 1/(2*(Hp))*np.ones(len(z))
        # P[z < Hp] = 0
        # P[z > Hp] = 0

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
cdef double gamma_xy(double zi_1, double a, double Dpara_0, double kBT, double Hs):
    """
    :return: gamma_plt.legend()x = gamma_y = 6πη(z)R : the gamma value for x and y trajectories dependant of z(t-dt).
    """
    cdef double gam_xy = (
        kBT / Dpara_0
        * ( 1 - (zi_1 / Hs)**2 )**(-1)
    )

    return gam_xy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double gamma_z(double zi_1, double a, double Dperp_0, double kBT, double H):
    """
    :return: gamma_z = 6πη(z)R : the gamma value for z trajectory dependant of z(t-dt).
    """
    cdef double gam_z = (
        kBT / Dperp_0
        * (1 - (zi_1/(H-a))**2)**(-1)
    )

    return gam_z


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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionXYi_cython(double xi_1, double zi_1, double rng, double dt, double a,
                               double Dpara_0, double kBT, double Hs):
    """
    :return: Position parallèle de la particule au temps suivant t+dt.
    """
    cdef double gamma = gamma_xy(zi_1, a, Dpara_0, kBT, Hs)
    cdef double xi = xi_1 + w(gamma, kBT) * rng * dt

    return xi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionZi_cython(double zi_1, double rng, double dt, double a,
                               double Dperp_0, double kBT, double H, double lB):
    """
    :return: Position perpendiculaire de la particule au temps suivant t+dt.
    """
    cdef double gamma = gamma_z(zi_1, a, Dperp_0, kBT, H)
    # PAS DE POTENTIEL !!!
    cdef double correction = - Dperp_0 * (2*zi_1/(H-a)**2)
    cdef double potentiel = kBT / lB #Potentiel gravitaire : Si lB->inf => pas de potentiel.
    cdef double zi = zi_1  + correction * dt + potentiel *dt /gamma +  w(gamma, kBT) * rng * dt
    # On ajoute le spurious drift dans la modélisation car F-Feq = +gg' = +D' gamma

    if zi < -H+a:
        zi = -2 * H -zi +2 * a
    if zi > H-a:
        zi = 2 * H - zi -2 * a

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
                                   double a,  double Dpara_0, double Dperp_0,
                                   double kBT, double H, double Hs, double lB):
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
            x = positionXYi_cython(x, z, random_gaussian()/sqrt(dt), dt, a, Dpara_0, kBT, Hs)
            y = positionXYi_cython(y, z, random_gaussian()/sqrt(dt), dt, a, Dpara_0, kBT, Hs)
            z = positionZi_cython(z, random_gaussian()/sqrt(dt), dt, a, Dperp_0, kBT, H, lB)

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
