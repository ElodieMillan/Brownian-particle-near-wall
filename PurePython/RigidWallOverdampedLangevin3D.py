# Élodie Millan
# June 2020
# Langevin equation 3D for a free particule close to a rigid wall without inertia and with weight.

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

# from OverdampedLangevin3D import Langevin3D
from InertialLangevin3D import InertialLangevin3D


class RigidWallOverdampedLangevin3D(InertialLangevin3D):  # , Langevin3D
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
        """
        Intern methode of RigidWallInertialLangevin3D class - gamma on x and y at time t-dt.

        :param zi_1: float - Perpendicular position by the wall z at (t - dt).

        :return: gamma_x = gamma_y = 6πη(z)R : the gamma value for x and y trajectories dependant of z(t-dt).
        """
        # Libchaber formula
        xi = self.R / (zi_1 + self.R)

        self.gamma_xy = (
                6
                * np.pi
                * self.R
                * self.eta
                * (
                        1
                        - 9 / 16 * xi
                        + 1 / 8 * xi ** 3
                        - 45 / 256 * xi ** 4
                        - 1 / 16 * xi ** 5
                )
                ** (-1)
        )
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
                (
                    (6 * zi_1 ** 2 + 2 * self.R * zi_1)
                    / (6 * zi_1 ** 2 + 9 * self.R * zi_1 + 2 * self.R ** 2)
                )
                ** (-1)
            )
        )

        # print("gamma_z = ", self.gamma_z)
        return self.gamma_z

    def _a(self, gamma):
        """
        Intern methode of RigidWallInertialLangevin3D class - white noise a = sqrt(k T gamma) at t-dt.

        :param gamma: the gamma value used (depends of the coordinate used).

        :return: The white noise a at the position z(t-dt) for a gamma value on x/y or z.
        """

        a = np.sqrt(2 * self.kb * self.T / gamma)

        return a

    def _PositionXi(self, xi_1, zi_1, rng, axis=None):
        """
        Intern methode of InertialLangevin3D class - Position of a Brownian particule inertial with rigid wall, at time t.

        :param xi_1: float - Position of the particule at (t - dt).
        :param zi_1: float - Perpendicular position by the wall z at (t - dt).
        :param rng: a random number for dBt/dt white noise.
        :param axis: The axis used : put "z" if z axis or None if x/y axis.

        :return: The position of the particule at time t.
        """

        if axis == "z":
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
               / ( (6 * zi_1 ** 2 + 9 * self.R * zi_1 + 2 * self.R ** 2) * (6*zi_1**2 + 2*self.R*zi_1) )
               * self.dt
               / gamma
            )

        else:
            gamma = self._gamma_xy(zi_1)
            elec = 0
            weight = 0
            correction = 0

        xi = xi_1 - weight + elec + correction + self._a(gamma) * rng * self.dt

        if axis == "z":
            if xi <= 0:
                xi = -xi

        return xi

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

        self.rngx = rngx
        self.rngy = rngy
        self.rngz = rngz

        x = np.zeros(self.Nt)
        y = np.zeros(self.Nt)
        z = np.zeros(self.Nt)

        # First values of trajectory compute with initial value.
        x[0] = self.x0[0]
        y[0] = self.x0[1]
        z[0] = self.x0[2]

        for i in tqdm(range(1, self.Nt)):
            x[i] = self._PositionXi(x[i - 1], z[i - 1], rngx[i])
            y[i] = self._PositionXi(y[i - 1], z[i - 1], rngy[i])
            z[i] = self._PositionXi(z[i - 1], z[i - 1], rngz[i], "z")

        self.x = x
        self.y = y
        self.z = z

        if output:
            return self.x, self.y, self.z

    ## SOME ANALYSIS FUNCTIONS

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
        for k, i in enumerate(tqdm(list_dt_c4)):
            if i == 0:
                c4[k] = 0
                continue
            c4[k] = (np.mean((position[i:] - position[:-i]) ** 4) - 3 * (
                np.mean((position[i:] - position[:-i]) ** 2)) ** 2) * 1 / (24)

        # --- Theory
        zth = np.linspace(1e-9, 10e-6, 300)
        if (axis == "z"):
            Di = self.kb * self.T / self._gamma_z(zth)
        else:
            Di = self.kb * self.T / self._gamma_xy(zth)

        P_eq_z = self.P_z_wall(zth, 4.8)
        P_eq_z = P_eq_z / np.trapz(P_eq_z, zth) #normalisation

        mean_Di_theo = np.trapz(Di * P_eq_z, zth)
        mean_Di2_theo = np.trapz(Di ** 2 * P_eq_z, zth)

        facteur_cumulant = (mean_Di2_theo - mean_Di_theo ** 2) / 2
        tth = np.linspace(np.min(self.t[list_dt_c4])/10,np.max(self.t[list_dt_c4])*10, 1000)

        if plot:
            plt.loglog(self.t[list_dt_c4], c4, "o", label=r"$\mathrm{Numerical}$")
            plt.plot(tth, facteur_cumulant * tth**2, "k-", label=r"$(\langle D_{\|, \mathrm{th}}^2 \rangle - \langle D_{\|, \mathrm{th}} \rangle^2)/2$")

            plt.xlabel(r"$\tau~(\mathrm{s})$", fontsize=15)
            plt.ylabel(r"$C^{(4)_"+axis+"}~(\mathrm{m}^4)$", fontsize=15)
            plt.axis([np.min(self.t[list_dt_c4]) / 5, np.max(self.t[list_dt_c4]) * 5, None, None])

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
            return self.t[list_dt_c4], c4



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
        tau_c = self.lB*self.R / self.D #equation (5.2.11) thèse Maxime :)
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
            binsPositions = binsPositions / np.sqrt(2 * D * dt * N_tau)
            pdf = hist / np.trapz(hist, binsPositions)

            z_theo = np.linspace(-5, 5, bins)
            PDFtheo = 1 / np.sqrt(2 * np.pi) * np.exp(-z_theo ** 2 / 2)

        if space == "wall":
            if axis == "x" or axis == "y" or axis == "z":
                hist, bin_edges = np.histogram(position[position < 3e-6], bins=bins, density=False)
                binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
                pdf = hist / np.trapz(hist, binsPositions)

                if axis == "z":
                    z_theo = np.linspace(1e-9, np.max(binsPositions*2), 10*bins)
                    p_theo = np.exp(- 4.8 * np.exp(-z_theo / self.lD) - z_theo / self.lB)
                    PDFtheo = p_theo / np.trapz(p_theo, z_theo)
            else:
                # Calcul de la théorie gaussienne qui marche plus
                dx_gauss = np.linspace(np.min(dX*10), np.max(dX*10), 1000)
                # std_gauss = np.sqrt(2*self.D* N_tau*self.dt)
                PDF_gauss = 1 / (np.sqrt(2 * np.pi)*std_num) * np.exp(-(dx_gauss/std_num) ** 2 / 2)  # Gaussian theory wrong
                PDF_gauss = PDF_gauss/np.trapz(PDF_gauss, dx_gauss)

                # Calcul de la PDF(dx) numerique
                hist, bin_edges = np.histogram(dX, bins=bins, density=False)
                binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
                pdf = hist / np.trapz(hist, binsPositions)
                binsPositions = binsPositions/std_num
                Label = r"$\tau = " + str(np.around(N_tau * self.dt, 4)) + "\mathrm{s}$"

                #Calcul de la théorie qui marche
                if (axis=="dz" and tau>=tau_c):
                    PDFtheo = self.P_deltaZ_longTime(dx_gauss, B)
                else:
                    PDFtheo = self.P_deltaXi_shortTime(dx_gauss, N_tau, axis, 1000, B)

                dx_theo = dx_gauss/std_num


        if plot:
            if space == "bulk":
                plt.semilogy(binsPositions, pdf, "o", markersize=4, label=r"$\mathrm{Numerical}$")
                plt.plot(z_theo, PDFtheo, "k-", label=r"$\mathrm{Theoritical}$")

                plt.xlabel(r"$" + Axis + "/ \sqrt{2D \Delta t} $")
                plt.ylabel(r"$P(" + Axis + ") $")

            if space == "wall":
                plt.semilogy(binsPositions, pdf, "o", label=r"$\mathrm{Numerical}$")

                if axis == "x" or axis == "y" or axis == "z":
                    if axis == "z":
                        plt.plot(z_theo, PDFtheo, "k-", label=r"$\mathrm{Theoritical}$")
                    plt.xlabel(r"$" + Axis + "(\mathrm{m})$", fontsize=15)
                    plt.ylabel(r"$P(" + Axis + ") ~ (\mathrm{m}^{-1})$", fontsize=15)
                else:
                    if (tau<tau_c):
                        plt.plot(dx_gauss/std_num, PDF_gauss, "r:", label=r"$\mathrm{Gaussian}$")
                    plt.plot(dx_theo, PDFtheo, "k-", label=r"$\mathrm{Theory}$")

                    plt.title(Label)
                    plt.xlabel(r"$" + Axis + "/ \sigma_\mathrm{num}$", fontsize=15)
                    plt.ylabel(r"$P(" + Axis + ")$", fontsize=15)

            ax = plt.gca()
            locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
            ax.yaxis.set_major_locator(locmaj)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.legend()
            plt.show()


    def P_z_wall(self, z, B):
        """

        :param A: Normalisation.
        :param B: 4.8 experimentally.

        :return: P_eq(z) near a wall.
        """
        if type(z) != np.ndarray:
            if z < 0:
                return 0
            return np.exp(-(B) * np.exp(- z / self.lD) - z / self.lB)

        P = np.exp(-(B) * np.exp(- z / self.lD) - z / self.lB)
        P[z < 0] = 0

        return P

    def Pz_PDeltaz(self, z, deltaz, A, B):
        # P(z)*P(z+Deltaz)
        PP = self.P_z_wall(z, A, B) * self.P_z_wall(z + deltaz, A, B)

        return PP

    def p_Deltaz_longTime(self, z, deltaz, A, B):
        # integrate of P(z)*P(z+Deltaz) on z
        PPP = np.trapz(self.Pz_PDeltaz(z, deltaz, A, B), z)

        return PPP

    def P_Deltaz_longTime(self, z, deltaz, B, lD, lB):
        A = 1
        PPPP = np.zeros(len(deltaz))
        for i in range(len(deltaz)):
            PPPP[i] = self._P_Deltaz_longTime(z, deltaz[i], A, B)
        A = 1 / np.trapz(PPPP, deltaz)

        return PPPP * A

    # Pour la théorie prêt d'un mur dur
    def P_Di(self, axis, bins, B):

        z = np.linspace(1e-10, 10e-6, bins)

        if (axis == "dz" or axis=="z"):
            Di = self.kb * self.T / self._gamma_z(z)

        else:
            Di = self.kb * self.T / self._gamma_xy(z)

        p_Di = Di * self.P_z_wall(z, B)
        p_Di = p_Di / np.trapz(p_Di, z)

        return Di, p_Di

    def _P_deltaXi_shortTime(self, dxi, N_tau, axis, bins, B):
        tau = N_tau*self.dt # tau from dx = x(t+tau)-x(t)
        Di, p_Di = self.P_Di(axis=axis, bins=bins, B=B)
        P = p_Di / np.sqrt(4 * np.pi * Di * tau) * np.exp(- dxi**2 / (4 * Di * tau))
        P = np.trapz(P,Di)

        return P

    def P_deltaXi_shortTime(self, list_dxi, N_tau, axis, bins, B):
        """

        :param list_dxi: List of dx=x(t+tau)-x(t).
        :param N_tau: Like tau = N_tau*dt, where dt is integration step.
        :param axis: Choose between "x", "y" or "z" axis.
        :param B:
        :return:
        """
        P = np.array([self._P_deltaXi_shortTime(dxi=i, N_tau=N_tau, axis=axis, bins=bins, B=B) for i in list_dxi])
        P = P/np.trapz(P, list_dxi)

        return P

    def _P_deltaZ_longTime(self, dz, B):

        z = np.linspace(1e-9, 10e-6, 500)
        dP = self.P_z_wall(z, B) * self.P_z_wall(z+dz, B)
        P = np.trapz(dP, z)

        return P

    def P_deltaZ_longTime(self, dz, B):
        Pdf = np.array([self._P_deltaZ_longTime(i,B) for i in dz])
        Pdf = Pdf/np.trapz(Pdf, dz)

        return Pdf



"""
FIN CLASSE
"""


def test():
    langevin3D = RigidWallOverdampedLangevin3D(
        dt=1 / 60, Nt=1000000, R=1.5e-6, rho=1050, x0=(0.0, 0.0, 1.0e-6)
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

if __name__ == '__main__':
    test()