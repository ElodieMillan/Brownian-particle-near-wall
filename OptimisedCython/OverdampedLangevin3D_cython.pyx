
import numpy as np
import matplotlib.pyplot as plt
cimport numpy as np
from libcpp cimport bool
import cython

DTYPE = np.float64

cdef class Langevin3D:
    """
    Brownian motion generation.
    """

    def __init__(self, DTYPE_t dt, unsigned long long int Nt, DTYPE_t R, DTYPE_t eta=0.001, DTYPE_t T=300., (DTYPE_t, DTYPE_t, DTYPE_t) x0=(0., 0., 0.)):
        """
        Constructor.

        :param dt: Time step [s].
        :param Nt: Number of time points.
        :param R: Radius of particule [m].
        :param eta: Fluid viscosity (default = 0.001 [Pa/s]).
        :param T: Temperature (default = 300 [k]).
        :param x0: Initial position of particule (default = (0,0,0) [m]).
        """
        self.dt = dt
        self.Nt = Nt
        self.R = R
        self.eta = eta
        self.T = T
        self.x0 = x0

        self.m = 0 #non inertial value
        self.kb = 1.38e-23
        self.gamma = 6 * np.pi * eta * R
        self.tau = 1
        self.a = np.sqrt((2 * self.kb * T) / self.gamma)
        self.D = (self.kb * T) / (self.gamma)
        self.t = np.arange(Nt) * dt
        self.x = np.zeros(self.Nt, dtype = DTYPE)
        self.y = np.zeros(self.Nt, dtype = DTYPE)
        self.z = np.zeros(self.Nt, dtype = DTYPE)
        self.list_dt_MSD = np.array([], dtype=np.int)


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function	
    cdef void trajectory(self):
        """
        Compute the trajectory of a Langevin3D particule.

        :param Nt : Number of point of times (default is the number give in the instance of the class).
        :return: return the x, y, z trajectory.
        """

        self.x =  + np.cumsum(
            self.a * np.random.default_rng().normal(0.0, np.sqrt(self.dt), size=self.Nt)
        )
        self.y = self.x0[1] + np.cumsum(
            self.a * np.random.default_rng().normal(0.0, np.sqrt(self.dt), size=self.Nt)
        )
        self.z = self.x0[2] + np.cumsum(
            self.a * np.random.default_rng().normal(0.0, np.sqrt(self.dt), size=self.Nt)
        )

    def plotTrajectory(self):
         """
         Plot the trajectory of the Langevin3D object.
         """
         plt.figure()
         plt.plot(self.t, self.x, color="blue", linewidth=0.8, label="x")
         plt.plot(self.t, self.y, color="red", linewidth=0.8, label="y")
         plt.plot(self.t, self.z, color="green", linewidth=0.8, label="z")
         plt.xlabel("Time [s]")
         plt.ylabel("Position [m]")
         plt.title("3D trajectory of free particle (using Cython)")
         plt.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
         plt.legend(loc="upper right")
         plt.tight_layout()
         plt.show()
    
    def MSD1D(self, axis, output=False, plot=False):
         """
         Compute the mean square displacement in 1 dimention.
    
         :param axis: The 1D trajectory to compute, "x" or "y" or "z".
         :param output: Boolean, if true function output MSD1D (default : false).
         :param plot: Boolean, if true plot MSD1D (default : false).
         :return: The mean square displacement in 1 dimension of the "axis" trajectory.
         """
         for i in range(len(str(self.Nt)) - 1):
             # Take just 10 points by decade.
             self.list_dt_MSD = np.concatenate(
                 (
                     self.list_dt_MSD,
                     np.arange(10 ** i, 10 ** (i + 1), 10 ** i, dtype=np.int),
                 )
             )
    
         if axis == "x":
             x = self.x
         elif axis == "y":
             x = self.y
         elif axis == "z":
             x = self.z
         else:
             raise ValueError("axis should be equal to 'x' or 'y' or 'z'")

         NumberOfMSDPoint = len(self.list_dt_MSD)
         cdef np.ndarray MSD = np.zeros(NumberOfMSDPoint, dtype = DTYPE)
         self.MSD = MSD
         for n, i in enumerate(self.list_dt_MSD):
             if i == 0:
                 self.MSD[n] = 0
                 continue
             self.MSD[n] = np.mean((x[i:] - x[0:-i]) ** 2)
    
         if plot:
             plt.figure()
             plt.loglog(
                 self.t[self.list_dt_MSD]/self.tau,
                 self.MSD,
                 "o",
                 color="red",
                 linewidth=0.8,
                 label="Experimental MSD" + axis,
             )
             plt.plot(
                 self.t[self.list_dt_MSD]/self.tau,
                 (2 * self.kb * self.T / self.gamma) * self.t[self.list_dt_MSD],
                 linewidth=0.8,
                 color="blue",
                 label="Non inertial theory : " + axis + " = 2D t",
             )

             if self.tau != 1:
                 plt.plot(self.t[self.list_dt_MSD] / self.tau,
                          (self.kb * self.T / (self.m)) * self.t[self.list_dt_MSD]**2,
                          linewidth=1.2,
                          linestyle="-",
                          color="green",
                          label="Max-Boltz theory: x = $(k_b T / m) t^2$")

                 plt.xlabel("Times t/$ \\tau $")
                 plt.text(np.max(self.t[self.list_dt_MSD]/self.tau)/10, np.min(self.MSD),
                          "$\\tau$ = {:.2e} s".format(self.tau))
             else:
                 plt.xlabel("Times t [s]")

             plt.ylabel("MSD 1D [m²]")
             plt.title("Mean square displacement 1D")
             plt.legend()
             plt.show()
    
         if output:
             return self.MSD
    
    
    def MSD3D(self, output=False, plot=False):
        """
        Compute the mean square displacement at 3D.
   
        :param output: Boolean, if true function output MSD1D (default : false).
        :param plot: Boolean, if true plot MSD1D (default : false).
        :return: The mean square displacement in 3 dimension of the trajectory.
        """
        self.MSD3 = (
            self.MSD1D("x", output=True,plot=False)
            + self.MSD1D("y", output=True,plot=False)
            + self.MSD1D(axis = "z", output=True,plot=False)
        )
    
        if plot:
            plt.figure()
            plt.loglog(
                self.t[self.list_dt_MSD]/self.tau,
                self.MSD3,
                color="red",
                linewidth=0.8,
                label="MSD3D ",
            )
            plt.plot(
                self.t[self.list_dt_MSD]/self.tau,
                (2 * 3 * self.kb * self.T / self.gamma) * self.t[self.list_dt_MSD],
                linewidth=0.8,
                label="Theory : x = 6D t",
            )
            plt.xlabel("Times t [s]")
            plt.ylabel("MSD 3D [m²]")
            plt.title("Mean square displacement 3D")
            plt.legend()
            plt.show()
    
        if output:
            return self.MSD3
    
    def speedDistribution1D(
        self, axis, nbTimesIntervalle=1, bins=50, output=False, plot=False
    ):
        """
        Compute the probability density function with Vx = [ x(t+ nbTimesIntervalle*dt) - x(t) ] /                 [ nbTimesIntervalle*dt ].
    
        :param axis: The 1D trajectory to compute, "x" or "y" or "z".
        :param nbTimesIntervalle: Number of times interval like x.dt (default = 1).
        :param bins : See numpy.histogram() documentation.
        :param output: Boolean, if true function output MSD1D (default : false).
        :param plot: Boolean, if true plot MSD1D (default : false).
        :return: Histogramme and bins positions.
        """
        if axis == "x":
            x = self.x
        elif axis == "y":
            x = self.y
        elif axis == "z":
            x = self.z
        else:
            raise ValueError("axis should be equal to 'x' or 'y' or 'z'")
   
        self.speed = (x[nbTimesIntervalle:] - x[:-nbTimesIntervalle]) / (
            nbTimesIntervalle * self.dt
        )
        hist, bin_edges = np.histogram(self.speed, bins=bins, density=True)
        binsPosition = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        if plot:
            plt.figure()
            plt.plot(
                binsPosition,
                hist,
                "o",
                label="Times interval = " + np.str(nbTimesIntervalle) + " dt",
            )
            plt.title("Probability density function 1D")
            plt.xlabel("Speeds " + axis + " $[m.s^{-1}]$")
            plt.ylabel("Density normalised $[m^{-1}.s]$")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.legend()
            plt.show()
    
        if output:
            return hist, binsPosition
    
    def dXDistribution1D(
        self, axis, nbTimesIntervalle=1, bins=50, output=False, plot=False
    ):
        """
        Compute the probability density function with dx = x(t+ nbTimesIntervalle*dt) - x(t)
   
        :param axis: The 1D trajectory to compute, "x" or "y" or "z".
        :param nbTimesIntervalle: Number of times interval like x.dt (default = 1).
        :param bins : See numpy.histogram() documentation.
        :param output: Boolean, if true function output MSD1D (default : false).
        :param plot: Boolean, if true plot MSD1D (default : false).
        :return: Histogramme and bins positions.
        """
        if axis == "x":
            x = self.x
        elif axis == "y":
            x = self.y
        elif axis == "z":
            x = self.z
        else:
            raise ValueError("axis should be equal to 'x' or 'y' or 'z'")
   
        self.dX = x[nbTimesIntervalle:] - x[:-nbTimesIntervalle]
        hist, bin_edges = np.histogram(self.dX, bins=bins, density=True)
        binsPosition = (bin_edges[:-1] + bin_edges[1:]) / 2
   
        if plot:
            plt.figure()
            plt.plot(
                binsPosition,
                hist,
                "o",
                label="Times interval = " + np.str(nbTimesIntervalle) + " dt",
            )
            plt.title("Probability density function 1D")
            plt.xlabel("$\Delta $" + axis + " $[m]$")
            plt.ylabel("Density normalised $[m^{-1}]$")
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.legend()
            plt.show()
   
        if output:
            return hist, binsPosition

def test():
    langevin3D = Langevin3D(0.01, 1000000, 1e-6)
    langevin3D.trajectory()

    langevin3D.plotTrajectory()
    langevin3D.MSD1D("x", plot=True)
    langevin3D.MSD1D("y", plot=True)
    langevin3D.MSD1D("z", plot=True)
    # langevin3D.MSD3D(plot=True)
    # #langevin3D.speedDistribution1D("x", 10, plot=True)
    # langevin3D.dXDistribution1D("x", 10, plot=True)
