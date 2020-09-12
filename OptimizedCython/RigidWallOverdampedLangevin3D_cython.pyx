"""
Élodie Millan
June 2020
Langevin equation 3D for a free particule close to a rigid wall without inertia and with weight.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt

from InertialLangevin3D_cython import InertialLangevin3D


class RigidWallOverdampedLangevin3D(InertialLangevin3D):
    def __init__(self, dt, Nt, R, rho, rhoF=1000.0, eta=0.001, T=300.0, x0=None):
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
            x0 = (0.0, 0.0, R)
        super().__init__(dt, Nt, R, rho, eta=eta, T=T, x0=x0)
        self.rhoF = rhoF
        self.lD = 70e-9  # Debay length
        self.g = 9.81  # m/s²
        self.m = rho * (4 / 3) * np.pi * R ** 3
        self.delta_m = (4 / 3) * np.pi * self.R ** 3 * (self.rho - self.rhoF)
        self.lB = (self.kb * self.T) / (self.delta_m * self.g)  # Boltzmann length

    def _gamma_xy(self, zi_1):
        """Interne methode :

        :param zi_1: float - Perpendicular position by the wall z at (t - dt).

        :return: gamma_x = gamma_y = 6πη(z)R : the gamma value for x and y trajectories dependant of z(t-dt).
        """
        # Libchaber formula
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

        return self.gamma_xy


    def _gamma_z(self, zi_1):
        """ Interne methode

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
                (
                    (6 * zi_1 ** 2 + 2 * self.R * zi_1)
                    / (6 * zi_1 ** 2 + 9 * self.R * zi_1 + 2 * self.R ** 2)
                )
                ** (-1)
            )
        )

        return self.gamma_z

    def trajectory(self, output=False):
        """
        :param output: Boolean - if true, return x, y, z (default : false).

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

        res = np.zeros((3, self.Nt))

        # First values of trajectory compute with initial value.
        res[0,0] = self.x0[0]
        res[1,0] = self.x0[1]
        res[2,0] = self.x0[2]

        res = np.asarray(trajectory_cython(self.Nt, rngx, rngy, rngz, res, self.delta_m, self.g, self.dt,
                                self.kb, self.T, self.lD, self.R, self.eta))

        self.x = res[0,:]
        self.y = res[1,:]
        self.z = res[2,:]

        if output:
            return self.x, self.y, self.z


"""
CYTHON We put all methode as function out of the class (Object as Cython is complex).
"""
cdef double pi = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double gamma_xy(double zi_1, double R, double eta):
    """
    :param zi_1: Particule position at time t-dt.
    :param R: Particule ray.
    :param eta: Fluide viscosity.
    
    :return: gamma_x = gamma_y = 6πη(z)R : the gamma value for x and y trajectories dependant of z(t-dt).
    """
    # Libchaber formula
    cdef double gam_xy = (
        6
        * pi
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

    return gam_xy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double gamma_z(double zi_1, double R, double eta):
    """
    :param zi_1: Particule position at time t-dt.
    :param R: Particule ray.
    :param eta: Fluide viscosity.
    
    :return: gamma_z = 6πη(z)R : the gamma value for z trajectory dependant of z(t-dt).
    """
    # Padé formula
    cdef double gam_z = (
        6
        * pi
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

    return gam_z

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double a(double gamma, double kb, double T):
    """
    :param gamma: Gamma on x/y or z (depend of the model).
    :param kb: Boltzmann constant.
    :param T: Temperature.
    
    :return: The white noise a at the position z(t-dt) for a gamma value on x/y or z.
    """
    cdef double noise = sqrt(2 * kb * T / gamma)  #**(1/2)

    return noise

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionXYi_cython(double xi_1, double zi_1, double rng, double dt,
                               double kb, double T, double R, double eta):
    """
    :param xi_1: Position of the particule at (t - dt).
    :param zi_1: Perpendicular position by the wall z at (t - dt).
    :param rng: a random number for dBt/dt white noise.
    :param zaxis: Test if z axis is used : put 1 if z axis or 0 if x/y axis (default value = 0).
    :param delta_m: Archimède force.
    :param g: Gravitation constant.
    :param dt: Times step.
    :param kb: Boltzmann constant.
    :param T: Temperature.
    :param lD: Debay length.
    :param R: Ray of the particule.
    :param eta: Fluid viscosity.
    
    :return: X or Y position of the particule at time t.
    """
    cdef double gamma = gamma_xy(zi_1, R, eta)
    cdef double xi = xi_1 + a(gamma, kb, T) * rng * dt

    return xi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionZi_cython(double zi_1, double rng, double delta_m, double g, double dt,
                               double kb, double T, double lD, double R, double eta):
    """
    :param xi_1: Position of the particule at (t - dt).
    :param zi_1: Perpendicular position by the wall z at (t - dt).
    :param rng: a random number for dBt/dt white noise.
    :param zaxis: Test if z axis is used : put 1 if z axis or 0 if x/y axis (default value = 0).
    :param delta_m: Archimède force.
    :param g: Gravitation constant.
    :param dt: Times step.
    :param kb: Boltzmann constant.
    :param T: Temperature.
    :param lD: Debay length.
    :param R: Ray of the particule.
    :param eta: Fluid viscosity.
    
    :return: Z position of the particule at time t.
    """
    cdef double gamma = gamma_z(zi_1, R, eta)
    cdef double weight = delta_m * g * dt / (gamma)
    cdef double elec = (
                (4 * kb * T) / lD
                * exp(-zi_1 / lD)
                * dt
                / gamma
    )
    cdef double correction = (
                kb * T
                * (42 * R * zi_1 ** 2 + 24 * R ** 2 * zi_1 + 4 * R ** 3)
                / ( (6 * zi_1 ** 2 + 9 * R * zi_1 + 2 * R ** 2) * (6*zi_1**2 + 2*R*zi_1) )
                * dt
                / gamma
    )
    cdef double zi = zi_1 - weight + elec + correction + a(gamma, kb, T) * rng * dt

    if zi <= 0: # because overdamped !!!
        zi = -zi


    return zi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[:,:] trajectory_cython(unsigned long int Nt,
                                   double[:] rngx,
                                   double[:] rngy,
                                   double[:] rngz,
                                   double[:,:] res,
                                   double delta_m, double g, double dt,
                                   double kb, double T, double lD, double R, double eta):
    """
    :param Nt: Number of time points.
    :param rngx: Random values (normal probability) on x trajectory.
    :param rngy: Random values (normal probability) on y trajectory.
    :param rngz: Random values (normal probability) on z trajectory.
    :param res: Results contain
    :param delta_m: Archimède force.
    :param g: Gravitation constant
    :param dt: Times step.
    :param kb: Boltzmann constant.
    :param T: Temperature.
    :param lD: Debay length.
    :param R: Ray of the particule.
    :param eta: Fluid viscosity.
    
    :return: X, Y, Z trajectory compute with Cython (faster).
    """
    cdef unsigned long int i

    for i in range(1, Nt):
        res[0,i] = positionXYi_cython(res[0,i - 1], res[2,i - 1], rngx[i], dt, kb, T, R, eta)
        res[1,i] = positionXYi_cython(res[1,i - 1], res[2,i - 1], rngy[i], dt, kb, T, R, eta)
        res[2,i] = positionZi_cython(res[2,i - 1], rngz[i], delta_m, g, dt, kb, T, lD, R, eta)
        
    return res


def test():
    langevin3D = RigidWallOverdampedLangevin3D(
        dt=1 / 60, Nt=1000000, R=1.5e-6, rho=1050, x0=(0.0, 0.0, 1.0e-6)
    )
    # langevin3D.trajectory()

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

if __name__ == '__main__':
    test()
