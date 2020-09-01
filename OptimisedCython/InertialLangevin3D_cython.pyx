
import numpy as np
import matplotlib.pyplot as plt
cimport numpy as np
import cython

from OverdampedLangevin3D_cython import Langevin3D
from OverdampedLangevin3D_cython cimport Langevin3D

DTYPE = np.float64 # C type equivalent at a DTYPE_t
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef class InertialLangevin3D(Langevin3D):
    """
    Brownian motion generation with inertia.
    """

    def __init__(self, DTYPE_t dt, unsigned long long int Nt, DTYPE_t R, DTYPE_t rho, DTYPE_t eta=0.001,
                 DTYPE_t T=300, (DTYPE_t, DTYPE_t, DTYPE_t) x0=(0, 0, 0)):
        """

        Constructor.

        :param dt: Time step [s].
        :param Nt: Number of time points.
        :param R: Radius of particule [m].
        :param rho: Volumic mass of the particule [kg/m³]
        :param eta: Fluid viscosity (default = 0.001 [Pa/s]).
        :param T: Temperature (default = 300 [k]).
        :param x0: Initial position of particule (default = (0,0,0) [m]).
        """
        super(InertialLangevin3D, self).__init__(dt, Nt, R, eta=eta, T=T, x0=x0)
        self.rho = rho
        self.m = rho * (4 / 3) * np.pi * R ** 3
        self.tau = self.m / self.gamma
        self.a = np.sqrt(2 * self.kb * self.T * self.gamma)  # Coef of the white noise
        self.b = 2 + dt / self.tau
        self.c = 1 + dt / self.tau
        self.x = np.zeros(self.Nt, dtype = DTYPE)
        self.y = np.zeros(self.Nt, dtype = DTYPE)
        self.z = np.zeros(self.Nt, dtype = DTYPE)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef DTYPE_t _PositionXi(self, DTYPE_t xi1, DTYPE_t xi2, DTYPE_t rng):
        """
        Intern methode of InertialLangevin3D class - Position of a Brownian particule at time t.

        :param xi1: Position of the particule at (t - dt).
        :param xi2: Position of the particule at (t - 2dt).
        :param rng: a random number for dBt white noise.
        :return: The position of the particule at time t.
        """
        xi = (
            (self.b / self.c * xi1)
            - (1 / self.c * xi2)
            + (self.a / self.c) * (self.dt ** 2 / self.m) * rng
        )

        return xi

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void trajectory(self):

        Nt = self.Nt


        cdef np.ndarray[DTYPE_t, ndim=1] rngx = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )
        cdef np.ndarray[DTYPE_t, ndim=1] rngy = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )
        cdef np.ndarray[DTYPE_t, ndim=1] rngz = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )

        cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros(self.Nt, dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(self.Nt, dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] z = np.zeros(self.Nt, dtype = DTYPE)


        # 2 first values of trajectory.
        x[0:2] = [self.x0[0], self.x0[0]]
        y[0:2] = [self.x0[1], self.x0[1]]
        z[0:2] = [self.x0[2], self.x0[2]]

        for i in range(2, Nt):

            x[i] = self._PositionXi(x[i - 1], x[i - 2], rngx[i])
            y[i] = self._PositionXi(y[i - 1], y[i - 2], rngy[i])
            z[i] = self._PositionXi(z[i - 1], z[i - 2], rngz[i])

        self.x = x
        self.y = y
        self.z = z


def test():
    langevin3D = InertialLangevin3D(1e-7, 2000000, 1e-6, 1050)

    langevin3D.trajectory()
    # langevin3D.plotTrajectory()
    #
    # langevin3D.MSD1D("x", plot=True)
    # langevin3D.MSD1D("y", plot=True)
    # langevin3D.MSD1D("z", plot=True)

    # MSDx = langevin3D.MSD1D("x", output=True)
    # MSDy = langevin3D.MSD1D("y", output=True)
    # MSDz = langevin3D.MSD1D("z", output=True)
    #
    # # ----- MSD 1D -----
    #
    # fig1 = plt.figure()
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     MSDx,
    #     color="red",
    #     linewidth=0.8,
    #     label="MSDx inertial",
    # )
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     MSDy,
    #     color="green",
    #     linewidth=0.8,
    #     label="MSDy inertial",
    # )
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     MSDz,
    #     color="blue",
    #     linewidth=0.8,
    #     label="MSDz inertial",
    # )
    # plt.plot(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     (2 * langevin3D.kb * langevin3D.T / langevin3D.gamma)
    #     * langevin3D.t[langevin3D.list_dt_MSD],
    #     color="black",
    #     linewidth=0.8,
    #     label="Non inertial theory : x = 2D t",
    # )
    # plt.xlabel("Times t/$ \tau $ [s]")
    # plt.ylabel("MSD 1D [m²]")
    # plt.title("Mean square displacement 1D")
    # plt.legend()
    # plt.show()
    #
    # # ----- MSD 3D -----
    #
    # MSD3D = langevin3D.MSD3D(output=True)
    # fig2 = plt.figure()
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     MSD3D,
    #     color="red",
    #     linewidth=0.8,
    #     label="Inertial MSD",
    # )
    # plt.plot(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     (6 * langevin3D.kb * langevin3D.T / langevin3D.gamma)
    #     * langevin3D.t[langevin3D.list_dt_MSD],
    #     color="black",
    #     linewidth=0.8,
    #     label="Non inertial theory : x = 6D t",
    # )
    # plt.xlabel("Times $ t/ \tau $")
    # plt.ylabel("MSD 3D [m²]")
    # plt.title("Mean square displacement 1D")
    # plt.legend()
    # plt.show()
    #
    # # langevin3D.speedDistribution1D("x", 10, plot=True)
    # langevin3D.dXDistribution1D("x", 10, plot=True)
