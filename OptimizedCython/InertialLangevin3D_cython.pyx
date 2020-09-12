# cython: infer_types=True

"""
Élodie Millan
June 2020
Langevin equation 3D bulk for a free particule with inertia.
"""

import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython

from OverdampedLangevin3D_cython import Langevin3D


class InertialLangevin3D(Langevin3D):
    def __init__(self, dt, Nt, R, rho, eta=0.001, T=300, x0=(0, 0, 0)):
        """

        :param dt: Time step [s].
        :param Nt: Number of time points.
        :param R: Radius of particule [m].
        :param rho: Volumic mass of the particule [kg/m³]
        :param eta: Fluid viscosity (default = 0.001 [Pa/s]).
        :param T: Temperature (default = 300 [k]).
        :param x0: Initial position of particule (default = (0,0,0) [m]).
        """
        super().__init__(dt, Nt, R, eta=eta, T=T, x0=x0)
        self.rho = rho

        self.m = np.float64(rho * (4 / 3) * np.pi * R ** 3)
        self.tau = np.float64(self.m / self.gamma)
        self.a = np.float64(np.sqrt(2 * self.kb * self.T * self.gamma))  # Coef of the white noise)
        self.b = np.float64(2 + dt / self.tau)
        self.c = np.float64(1 + dt / self.tau)


    def trajectory(self, output=False, Nt=None):

        if Nt == None:
            Nt = self.Nt

        rngx = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )
        rngy = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )
        rngz = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )

        res = np.zeros((3,Nt))


        ## Comment : it's faster to give an array to cython than create a new type on cython
        #            because you can't buffer the exact type & shape of the np.array

        # 2 first values
        res[0,0:2] = self.x0[0]
        res[1,0:2] = self.x0[1]
        res[2,0:2] = self.x0[2]


        res = trajectory_cython(self.Nt,
                                rngx,
                                rngy,
                                rngz,
                                res,
                                self.a,
                                self.b,
                                self.c,
                                self.dt,
                                self.m)

        self.x = res[0,:]
        self.y = res[1,:]
        self.z = res[2,:]
        if output:
            return self.x, self.y, self.z

"""
CYTHON 
"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dtype_t positionXi_cython(dtype_t a, dtype_t b, dtype_t c, dtype_t dt, dtype_t m, dtype_t xi1, dtype_t xi2, dtype_t rng):
    cdef dtype_t xi = (
        (b / c * xi1)
        - (1 / c * xi2)
        + (a / c) * (dt ** 2 / m) * rng
    )
    return xi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[dtype_t, ndim=2] trajectory_cython(long int Nt,
                                                   np.ndarray[dtype_t, ndim=1] rngx,
                                                   np.ndarray[dtype_t, ndim=1] rngy,
                                                   np.ndarray[dtype_t, ndim=1] rngz,
                                                   np.ndarray[dtype_t, ndim=2] res,
                                                   dtype_t a, dtype_t b, dtype_t c, dtype_t dt, dtype_t m):

    for i in range(2, Nt):
        res[0,i] = positionXi_cython(a, b, c, dt, m, res[0,i - 1], res[0,i - 2], rngx[i])
        res[1,i] = positionXi_cython(a, b, c, dt, m, res[1,i - 1], res[1,i - 2], rngy[i])
        res[2,i] = positionXi_cython(a, b, c, dt, m, res[2,i - 1], res[2,i - 2], rngz[i])

    return res



def test():
    langevin3D = InertialLangevin3D(1e-7, 5000000, 1e-6, 1050)

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

if __name__ == '__main__':
    test()