# Élodie Millan
# June 2020
# Langevin equation 3D for a free particule close to a rigid wall with inertia and weight.

import numpy as np
import matplotlib.pyplot as plt
#from OverdampedLangevin3D import Langevin3D
from InertialLangevin3D import InertialLangevin3D


class RigidWallInertialLangevin3D(InertialLangevin3D): #, Langevin3D
    def __init__(self, dt, Nt, R, rho, rhoF=1000., eta=0.001, T=300., x0=None):
        """

        :param dt: float - Time step [s].
        :param Nt: int - Number of time points.
        :param R: float - Radius of particule [m].
        :param rho: float - Volumic mass of the particule [kg/m³].
        :param rhoF: float - Volumic mass of the fluid [kg/m³] (DEFAULT = 1000 [kg/m³]).
        :param eta: float - Fluid viscosity (DEFAULT = 0.001 [Pa/s]).
        :param T: float - Temperature (DEFAULT = 300 [k]).
        :param x0: array float - Initial position of particule (DEFAULT = (0,0,R) [m]).
        """
        if x0 == None:
            x0 = (0., 0., R)
        super().__init__(dt, Nt, R, rho, eta=eta, T=T, x0=x0)
        self.rhoF = rhoF
        self.g = 9.81  # m/s²
        self.m = rho * (4 / 3) * np.pi * R ** 3
        self.delta_m = self.m - (4 / 3) * np.pi * self.R ** 3 * self.rhoF

    def _gamma_xy(self, zi_1):
        """
        Intern methode of RigidWallInertialLangevin3D class - gamma on x and y at time t-dt.

        :param zi_1: float - Perpendicular position by the wall z at (t - dt).

        :return: gamma_x = gamma_y = 6πη(z)R : the gamma value for x and y trajectories dependant of z(t-dt).
        """
        self.gamma_xy = (
            6
            * np.pi
            * self.R
            * self.eta
            * (
                1
                - ((9 * self.R) / (16 * (zi_1 + self.R)))
                + (self.R / (8 * (zi_1 + self.R))) ** 3
                - (45 * self.R / (256 * (zi_1 + self.R))) ** 4
                - (self.R / (16 * (zi_1 + self.R))) ** 5
            )
            ** (-1)
        )
        # print("gamma_xy = ", self.gamma_xy)
        return self.gamma_xy

    def _gamma_z(self, zi_1):
        """
        Intern methode of RigidWallInertialLangevin3D class - gamma on z at time t-dt.

        :param zi_1: float - Perpendicular position by the wall z at (t - dt).

        :return: float - gamma_z = 6πη(z)R : the gamma value for z trajectory dependant of z(t-dt).
        """
        # Padé formula
        self.gamma_z = (
            6
            * np.pi
            * self.R
            * self.eta
            * (
               (6 * zi_1 ** 2 + 2 * self.R * zi_1)
               / (6 * zi_1 ** 2 + 9 * self.R * zi_1 + 2 * self.R ** 2)
            )
            ** (-1)
        )
        # print("gamma_z = ", self.gamma_z)
        return self.gamma_z

    def _a(self, zi_1, gamma):
        """
        Intern methode of RigidWallInertialLangevin3D class - white noise a = sqrt(k T gamma) at t-dt.

        :param zi_1: float - Perpendicular position by the wall z at (t - dt).
        :param gamma: the gamma value used (depends of the coordinate used).

        :return: The white noise a at the position z(t-dt) for a gamma value on x/y or z.
        """
        # print("gamma = ", gamma)
        a = np.sqrt(2 * self.kb * self.T * gamma)

        return a

    def _PositionXi(self, xi_1, xi_2, zi_1, rng, axis=None):
        """
        Intern methode of InertialLangevin3D class - Position of a Brownian particule inertial with rigid wall, at time t.

        :param xi_1: float - Position of the particule at (t - dt).
        :param xi_2: float - Position of the particule at (t - 2dt).
        :param zi_1: float - Perpendicular position by the wall z at (t - dt).
        :param rng: a random number for dBt/dt white noise.
        :param axis: The axis used : put "z"
        :return:
        :param xi1: Position of the particule at (t - dt).
        :param xi2: Position of the particule at (t - 2dt).


        :return: The position of the particule at time t.
        """

        if axis == "z":
            gamma = self._gamma_z
            weight = -(self.delta_m / self.m) * self.g * self.dt ** 2
            # print(self.delta_m)
        else:
            gamma = self._gamma_xy
            weight = 0
        b = 2 + (self.dt * gamma(zi_1) / self.m)
        c = 1 + (self.dt * gamma(zi_1) / self.m)

        xi = (
            ((b / c) * xi_1)
            - ((1 / c) * xi_2)
            + (self._a(zi_1, gamma(zi_1)) / c) * (self.dt ** 2 / self.m) * rng
            + weight/c
        )
        #print("rdmterme = {}".format((self._a(zi_1, gamma(zi_1)) / c) * (self.dt ** 2 / self.m) * rng))

        return xi

    def trajectory(self, output=False):

        rngx = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )
        rngy = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )
        rngz = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )

        x = np.zeros(self.Nt)
        y = np.zeros(self.Nt)
        z = np.zeros(self.Nt)

        # 2 first values of trajectory compute with random trajectory.
        # x[0:2], y[0:2], z[0:2] = super().trajectory(output=True, Nt=2)
        x[0:2] = np.array([self.x0[0], self.x0[0]])
        y[0:2] = np.array([self.x0[1], self.x0[1]])
        z[0:2] = np.array([self.x0[2], self.x0[2]])

        for i in range(2, self.Nt):
            x[i] = self._PositionXi(x[i - 1], x[i - 2], z[i - 1], rngx[i])
            y[i] = self._PositionXi(y[i - 1], y[i - 2], z[i - 1], rngy[i])
            z[i] = self._PositionXi(z[i - 1], z[i - 2], z[i - 1], rngz[i], "z")

        self.x = x
        self.y = y
        self.z = z

        if output:
            return self.x, self.y, self.z


if __name__ == "__main__":
    langevin3D = RigidWallInertialLangevin3D(1e-7, 1000000, 1e-6, 1040)

    langevin3D.trajectory()
    langevin3D.plotTrajectory()
    MSDx = langevin3D.MSD1D("x", output=True)
    MSDy = langevin3D.MSD1D("y", output=True)
    MSDz = langevin3D.MSD1D("z", output=True)

    # ----- MSD 1D -----

    fig1 = plt.figure()
    plt.loglog(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        MSDx,
        color="red",
        linewidth=0.8,
        label="MSDx inertial",
    )
    plt.loglog(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        MSDy,
        color="green",
        linewidth=0.8,
        label="MSDy inertial",
    )
    plt.loglog(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        MSDz,
        color="blue",
        linewidth=0.8,
        label="MSDz inertial",
    )
    plt.plot(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        (2 * langevin3D.kb * langevin3D.T / langevin3D.gamma)
        * langevin3D.t[langevin3D.list_dt_MSD],
        color="black",
        linewidth=0.8,
        label="Non inertial theory : x = 2D t",
    )
    plt.xlabel("Times t/$ \tau $ [s]")
    plt.ylabel("MSD 1D [m²]")
    plt.title("Mean square displacement 1D")
    plt.legend()
    plt.show()

    # ----- MSD 3D -----

    MSD3D = langevin3D.MSD3D(output=True)
    fig2 = plt.figure()
    plt.loglog(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        MSD3D,
        color="red",
        linewidth=0.8,
        label="Inertial MSD",
    )
    plt.plot(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        (6 * langevin3D.kb * langevin3D.T / langevin3D.gamma)
        * langevin3D.t[langevin3D.list_dt_MSD],
        color="black",
        linewidth=0.8,
        label="Non inertial theory : x = 6D t",
    )
    plt.xlabel("Times $ t/ \tau $")
    plt.ylabel("MSD 3D [m²]")
    plt.title("Mean square displacement 1D")
    plt.legend()
    plt.show()

    # langevin3D.speedDistribution1D("x", 10, plot=True)
    langevin3D.dXDistribution1D("x", 10, plot=True)
