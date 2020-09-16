"""
Élodie Millan
June 2020
Langevin equation 3D for a free particule close to a rigid wall with inertia and with weight.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt

from RigidWallOverdampedLangevin3D_cython import RigidWallOverdampedLangevin3D

class RigidWallInertialLangevin3D( RigidWallOverdampedLangevin3D ):
    def __init__(self, dt, Nt, R, rho, rhoF=1000.0, eta=0.001, T=300.0, x0=None):
        """

        :param dt: double - Time step [s].
        :param Nt: int - Number of time points.
        :param R: double - Radius of particule [m].
        :param rho: double - Volumic mass of the particule [kg/m³].
        :param rhoF: double - Volumic mass of the fluid [kg/m³] (DEFAULT = 1000 [kg/m³]).
        :param eta: double - Fluid viscosity (DEFAULT = 0.001 [Pa/s]).
        :param T: double - Temperature (DEFAULT = 300 [k]).
        :param x0: array double - Initial position of particule (DEFAULT = (0,0,R) [m]).
        """
        if x0 == None:
            x0 = (0.0, 0.0, R)
        super().__init__(dt, Nt, R, rho, eta=eta, T=T, x0=x0)

        self.gamma_mean = 6 * np.pi * eta * R # average of gamma
        self.tau_mean = self.m / self.gamma_mean # average of tau

    @cython.profile(True)
    def trajectory(self, output=False):
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

        res = np.zeros((3, self.Nt))

        # 2 first values of trajectory
        res[0,0:2] = [self.x0[0],self.x0[0]]
        res[1,0:2] = [self.x0[1], self.x0[1]]
        res[2,0:2] = [self.x0[2], self.x0[2]]

        res = np.asarray(trajectory_cython(self.Nt, rngx, rngy, rngz, res, self.delta_m, self.g, self.dt,
                                           self.kb, self.T, self.lD, self.R, self.eta, self.m))
        self.x = res[0,:]
        self.y = res[1,:]
        self.z = res[2,:]

        if output:
            return self.x, self.y, self.z


"""
CYTHON We put all methode as function out of the class (Object as Cython is complex).
"""
cdef double pi = np.pi
cdef double gam_xy
cdef double gam_z
cdef double noise
cdef double gamma
cdef double weight
cdef double elec
cdef double b
cdef double c
cdef double xi
cdef double zi


@cython.profile(True)
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
    gam_xy = (
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

@cython.profile(True)
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
    gam_z = (
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

@cython.profile(True)
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
    noise = sqrt(2 * kb * T * gamma)

    return noise

# cdef double RNG():
#     cdef double rng =

@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionXYi_cython(double xi_1, double xi_2, double zi_1, double rng, double dt,
                               double kb, double T, double R, double eta, double m):

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
    gamma = gamma_xy(zi_1, R, eta)
    b = 2 + dt * gamma / m
    c = 1 + dt * gamma / m
    xi = 1/c * ( b * xi_1 - xi_2 + a(gamma, kb, T) * (dt ** 2 / m) * rng)

    return xi

@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double positionZi_cython(double zi_1, double zi_2, double rng, double delta_m, double g, double dt,
                               double kb, double T, double lD, double R, double eta, double m):
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
    gamma = gamma_z(zi_1, R, eta)
    weight = -(delta_m / m) * g * dt ** 2
    elec = (4 * kb * T) / (lD * m) * exp(-zi_1 / lD) * dt ** 2
    b = 2 + dt * gamma / m
    c = 1 + dt * gamma / m

    zi = 1/c * ( b * zi_1 - zi_2 + weight + elec + a(gamma, kb, T) * (dt ** 2 / m) * rng)

    return zi

@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[:,:] trajectory_cython(unsigned long int Nt,
                                   double[:] rngx,
                                   double[:] rngy,
                                   double[:] rngz,
                                   double[:,:] res,
                                   double delta_m, double g, double dt,
                                   double kb, double T, double lD, double R, double eta, double m):
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
    for i in range(2, Nt):
        res[0,i] = positionXYi_cython(res[0,i - 1], res[0,i - 2], res[2,i - 1], rngx[i], dt, kb, T, R, eta, m)
        res[1,i] = positionXYi_cython(res[1,i - 1], res[1,i - 2], res[2,i - 1], rngy[i], dt, kb, T, R, eta, m)
        res[2,i] = positionZi_cython(res[2,i - 1], res[2,i - 2], rngz[i], delta_m, g, dt, kb, T, lD, R, eta, m)

    return res

def test():
    langevin3D = RigidWallInertialLangevin3D(
        dt=1e-6, Nt=1000000, R=1.5e-6, rho=2500, x0=(0.0, 0.0, 1e-7)
    )
    langevin3D.trajectory()

    # langevin3D.plotTrajectory()
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
    #     (2 * langevin3D.kb * langevin3D.T / langevin3D.gamma_mean)
    #     * langevin3D.t[langevin3D.list_dt_MSD],
    #     color="black",
    #     linewidth=0.8,
    #     label="Non inertial theory : x = 2D t",
    # )
    # plt.xlabel("Times t [s]")
    # plt.ylabel("MSD 1D [m²]")
    # plt.title("Mean square displacement 1D")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    test()