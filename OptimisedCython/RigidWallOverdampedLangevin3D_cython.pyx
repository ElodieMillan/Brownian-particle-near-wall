# Élodie Millan
# June 2020
# Langevin equation 3D for a free particule close to a rigid wall without inertia and with weight.

import numpy as np
cimport numpy as np
import cython

from InertialLangevin3D_cython cimport InertialLangevin3D

DTYPE = np.float64 # C type equivalent at a DTYPE_t
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef class RigidWallOverdampedLangevin3D(InertialLangevin3D):
    """
    Brownian motion generation with inertia near a rigid wall (on z axis).
    """
    def __init__(self, DTYPE_t dt, unsigned long long int Nt, DTYPE_t R, DTYPE_t rho, DTYPE_t rhoF=1000.0,
                 DTYPE_t eta=0.001, DTYPE_t T=300.0, (DTYPE_t, DTYPE_t, DTYPE_t) x0=(0, 0, 1e-6)):
        """

        Constructor.

        :param dt: float - Time step [s].
        :param Nt: int - Number of time points.
        :param R: float - Radius of particule [m].
        :param rho: float - Volumic mass of the particule [kg/m³].
        :param rhoF: float - Volumic mass of the fluid [kg/m³] (DEFAULT = 1000 [kg/m³]).
        :param eta: float - Fluid viscosity (DEFAULT = 0.001 [Pa/s]).
        :param T: float - Temperature (DEFAULT = 300 [k]).
        :param x0: array float - Initial position of particule (DEFAULT = (0,0,1e-6) [m]).
        """
        super(RigidWallOverdampedLangevin3D, self).__init__(dt, Nt, R, rho, eta=eta, T=T, x0=x0)
        self.rhoF = rhoF
        self.lD = 70e-9  # Debay length
        self.g = 9.81  # m/s²
        self.m = rho * (4 / 3) * np.pi * R ** 3
        self.delta_m = (4 / 3) * np.pi * self.R ** 3 * (self.rho - self.rhoF)
        self.lB = (self.kb * self.T) / (self.delta_m * self.g)  # Boltzmann length
        self.x = np.zeros(self.Nt, dtype = DTYPE)
        self.y = np.zeros(self.Nt, dtype = DTYPE)
        self.z = np.zeros(self.Nt, dtype = DTYPE)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def trajectory(self):
        """

        :param output: Boolean, if true function output x, y, z (default : false).

        :return: return the x, y, z trajectory.
        """
        rngx = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )
        rngy = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )
        rngz = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )

        # self.rngx = rngx
        # self.rngy = rngy
        # self.rngz = rngz

        x = np.zeros(self.Nt)
        y = np.zeros(self.Nt)
        z = np.zeros(self.Nt)

        # First values of trajectory compute with initial value.
        x[0] = self.x0[0]
        y[0] = self.x0[1]
        z[0] = self.x0[2]

        cdef np.ndarray_t results = compute(x, y, z, rngx, rngy, rngz)

        self.x = results[]
        self.y = y
        self.z = z

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef np.ndarray_t compute(np.ndarray_t x, np.ndarray_t y, np.ndarray_t z,
                          np.ndarray_t rngx, np.ndarray_t rngy, np.ndarray_t rngz):
    for i in range(1, self.Nt):
        x[i] = PositionXi(x[i - 1], z[i - 1], rngx[i])
        y[i] = PositionXi(y[i - 1], z[i - 1], rngy[i])
        z[i] = PositionZi(z[i - 1], z[i - 1], rngz[i])

    cdef np.ndarray results = np.array([x, y, z])

    return results

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_t gamma_xy(DTYPE_t zi_1, DTYPE_t R, DTYPE_t eta):
    """
    Gamma on x and y at time t-dt.

    :param zi_1: float - Perpendicular position by the wall z at (t - dt).
    :param R: float - Ray of particles.
    :param eta: float - Fluid viscosity (DEFAULT = 0.001 [Pa/s]).

    :return: gamma_x = gamma_y = 6πη(z)R : the gamma value for x and y trajectories dependant of z(t-dt).
    """
    # Libchaber formula
    cdef DTYPE_t gamma_xy = (
        6
        * np.pi
        * R
        * eta
        * (
            1
            - ((9 * R) / (16 * (zi_1 + R)))
            + (R / (8 * (zi_1 + R))) ** 3
            - (45 * R / (256 * (zi_1 + R))) ** 4
            - (R / (16 * (zi_1 + R))) ** 5
        )
        ** (-1)
    )

    return gamma_xy

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_t gamma_z(DTYPE_t zi_1, DTYPE_t R, DTYPE_t eta):
    """
    Gamma on z at time t-dt.

    :param zi_1: float - Perpendicular position by the wall z at (t - dt).
    :param R: float - Ray of particles.
    :param eta: float - Fluid viscosity (DEFAULT = 0.001 [Pa/s]).

    :return: float - gamma_z = 6πη(z)R : the gamma value for z trajectory dependant of z(t-dt).
    """
    # Padé formula
    cdef DTYPE_t gamma_z = (
        6
        * np.pi
        * R
        * eta
        * (
            (
                (6 * zi_1 ** 2 + 2 * R * zi_1)
                / (6 * zi_1 ** 2 + 9 * R * zi_1 + 2 * R ** 2)
            )
            ** (-1)
        )
    )

    return gamma_z

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_t a(DTYPE_t gamma, DTYPE_t kb, DTYPE_t T):
    """
    Intern methode of RigidWallInertialLangevin3D class - white noise a = sqrt(k T gamma) at t-dt.

    :param gamma: the gamma value used (depends of the coordinate used).
    :param kb: float - Boltzmann constant.
    :param T: float - Temperature.

    :return: The white noise a at the position z(t-dt) for a gamma value on x/y or z.
    """

    cdef DTYPE_t a = np.sqrt(2 * kb * T / gamma)

    return a

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_t PositionXi(DTYPE_t xi_1, DTYPE_t zi_1, DTYPE_t rng,
                        DTYPE_t dt, DTYPE_t R, DTYPE_t eta, DTYPE_tkb, DTYPE_t T):
    """
    Intern methode of InertialLangevin3D class - Position x or y of a Brownian particule inertial with rigid wall, at time t.

    :param xi_1: float - Position of the particule at (t - dt).
    :param xi_2: float - Position of the particule at (t - 2dt).
    :param zi_1: float - Perpendicular position by the wall z at (t - dt).
    :param rng: a random number for dBt/dt white noise.
    :param dt: float - times scales.
    :param R: float - Ray of particles.
    :param eta: float - Fluid viscosity (DEFAULT = 0.001 [Pa/s]).
    :param kb: float - Boltzmann constant.
    :param T: float - Temperature.

    :return: The position of the particule at time t.
    """
    gamma = gamma_xy(zi_1, R, eta)

    xi = xi_1 + a(gamma, kb, T) * rng * dt

    return xi

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef DTYPE_t _PositionZi(self, DTYPE_t xi_1, DTYPE_t zi_1, DTYPE_t rng):
    """
    Intern methode of InertialLangevin3D class - Position x or y of a Brownian particule inertial with rigid wall, at time t.

    :param xi_1: float - Position of the particule at (t - dt).
    :param xi_2: float - Position of the particule at (t - 2dt).
    :param zi_1: float - Perpendicular position by the wall z at (t - dt).
    :param rng: a random number for dBt/dt white noise.

    :return: The position of the particule at time t.
    """

    gamma = self._gamma_z(zi_1)
    weight = self.delta_m * self.g * self.dt / (gamma)
    elec = (
        (4 * self.kb * self.T)
        / (self.lD)
        * np.exp(-zi_1 / self.lD)
        * self.dt
        / gamma
    )
    correction = (
        self.kb
        * self.T
        * (42 * self.R * zi_1 ** 2 + 24 * self.R ** 2 * zi_1 + 4 * self.R ** 3)
        / ((6 * zi_1 ** 2 + 9 * self.R * zi_1 + 2 * self.R ** 2) * (6*zi_1**2 + 2*self.R*zi_1) )
        * self.dt
        / gamma
    )

    xi = xi_1 - weight + elec + correction + self._a(gamma) * rng * self.dt

    if xi <= 0:
        xi = -xi

    return xi

# cdef DTYPE_t _PositionXi(self, DTYPE_t xi_1, DTYPE_t zi_1, DTYPE_t rng):
#     """
#     Intern methode of InertialLangevin3D class - Position x or y of a Brownian particule inertial with rigid wall, at time t.
#
#     :param xi_1: float - Position of the particule at (t - dt).
#     :param xi_2: float - Position of the particule at (t - 2dt).
#     :param zi_1: float - Perpendicular position by the wall z at (t - dt).
#     :param rng: a random number for dBt/dt white noise.
#
#     :return: The position of the particule at time t.
#     """
#
#     if axis == 0:
#         gamma = self._gamma_z(zi_1)
#         weight = self.delta_m * self.g * self.dt / (gamma)
#         elec = ((4 * self.kb * self.T)
#                 / (self.lD)
#                 * np.exp(-zi_1 / self.lD)
#                 * self.dt
#                 / gamma
#                 )
#         correction = (
#             self.kb
#             * self.T
#             * (42 * self.R * zi_1 ** 2 + 24 * self.R ** 2 * zi_1 + 4 * self.R ** 3)
#             / ((6 * zi_1 ** 2 + 9 * self.R * zi_1 + 2 * self.R ** 2) * (6*zi_1**2 + 2*self.R*zi_1) )
#             * self.dt
#             / gamma
#         )
#
#     else:
#         gamma = self._gamma_xy(zi_1)
#         elec = 0
#         weight = 0
#         correction = 0
#
#     xi = xi_1 - weight + elec + correction + self._a(gamma) * rng * self.dt
#
#     if axis == 0:
#         if xi <= 0:
#             xi = -xi
#
#     return xi



def test():

    langevin3D = RigidWallOverdampedLangevin3D(
        dt=1 / 60, Nt=10000, R=1.5e-6, rho=1050, x0=(1e-6, 1e-6, 1.0e-6)
    )
    langevin3D.trajectory()
    # langevin3D.plotTrajectory()
    #
    # MSDx = langevin3D.MSD1D("x", output=True)
    # MSDy = langevin3D.MSD1D("y", output=True)
    # MSDz = langevin3D.MSD1D("z", output=True)
    #
    # # ----- MSD 1D -----
    #
    # fig1 = plt.figure()
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD],
    #     MSDx,
    #     color="red",
    #     linewidth=0.8,
    #     label="MSDx inertial",
    # )
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD],
    #     MSDy,
    #     color="green",
    #     linewidth=0.8,
    #     label="MSDy inertial",
    # )
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD],
    #     MSDz,
    #     color="blue",
    #     linewidth=0.8,
    #     label="MSDz inertial",
    # )
    # plt.plot(
    #     langevin3D.t[langevin3D.list_dt_MSD],
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
    # plt.plot(langevin3D.t, langevin3D.z * 1e6)
    # plt.show()
