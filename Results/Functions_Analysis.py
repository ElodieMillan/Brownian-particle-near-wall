import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


kBT = 4.0e-21

def plot_traj(t,Rs):
    plt.plot(t, Rs[:,0], "-", label=r"$x$", lw=0.5)
    plt.plot(t, Rs[:,1], "-", label=r"$y$", lw=0.5)
    plt.plot(t, Rs[:,2], "-", label=r"$z$", lw=0.5)

    plt.xlabel(r"$t$ $(\mathrm{s})$", fontsize=15)
    plt.ylabel(r"$\mathrm{positions}~(\mathrm{m})$", fontsize=15)
    plt.legend()
    plt.show()

    plt.show()


def MSD(axis, Rs, dt, Nt, D_mean=None, lD=None, lB=None, speed_drift=None, space=None, plot=True, output=False):
    """

    Parameters
    ----------
    axis : "x", "y" or "z"
    Rs : datas to analyse
    dt : time step
    D : diffusion coeficient
    lD : Debye Lenght
    lB : Boltzman lenght
    speed_drift : value of drift speed
    space : "wall" or "bulk" domaine
    plot : True if plot is needed
    output : if True : return {MSD_axis, tau} (the delta t of displacements)

    -------

    """
    time = np.arange(0, Nt, dt)

    if axis == "x":
        position = Rs[:, 0]
    elif axis == "y":
        position = Rs[:, 1]
    elif axis == "z":
        position = Rs[:, 2]
    else:
        raise ValueError("WRONG AXIS : choose between 'x', 'y' and 'z' !")

    list_dt_MSD = np.array([], dtype=int)
    for i in range(len(str(Nt)) - 1):
        # Take just 10 points by decade.
        list_dt_MSD = np.concatenate(
            (
                list_dt_MSD,
                np.arange(10 ** i, 10 ** (i + 1), 10 ** i, dtype=int),
            )
        )

    # ----- Theoritical -----
    if D_mean != None:
        msd_theo_bulk = 2 * D_mean * time[list_dt_MSD]
    else:
        msd_theo_bulk = np.zeros(len(list_dt_MSD))
    if axis=="x" and speed_drift!=None:
        msd_x_drift = speed_drift**2 * time[list_dt_MSD]**2
    # -----------------------

    NumberOfMSDPoint = len(list_dt_MSD)
    msd = np.zeros(NumberOfMSDPoint)
    for k, i in enumerate(tqdm(list_dt_MSD)):
        if i == 0:
            msd[k] = 0
            continue
        msd[k] = np.mean((position[i:]-position[:-i])**2)

    if plot:
        plt.loglog(time[list_dt_MSD], msd, "o", label="Numerical")

        if axis=="x" and speed_drift != None:
            if space=="bulk":
                plt.plot(time[list_dt_MSD], (msd_theo_bulk + msd_x_drift), "k-", label= r"$2 D_0 \tau + u^2 \tau^2 $")
            elif space=="wall":
                plt.plot(time[list_dt_MSD], msd_theo_bulk + msd_x_drift, "k-",
                         label=r"$2 \langle D_x \rangle \tau + \langle u \rangle _z ^2 \tau^2 $")
                plt.plot(time[list_dt_MSD], msd_x_drift, ":", label=r"$\langle u \rangle _z ^2 \tau^2 $")
                plt.plot(time[list_dt_MSD], msd_theo_bulk, ":", label=r"$2 \langle D_x \rangle \tau $")

        else:
            if space=="bulk":
                plt.plot(time[list_dt_MSD], msd_theo_bulk, "k-", label=r"$2D_0 \tau$")
            elif space=="wall":
                plt.plot(time[list_dt_MSD], msd_theo_bulk, "k-", label=r"$2 \langle D_{} \rangle _z \tau$".format(axis))
                if axis=="z":
                    dz = np.linspace(-2.5e-5, 2.5e-5, 1000)
                    P_dz = P_Deltaz_longTime(dz, 4., lD, lB)
                    mean_dz_square = np.trapz(dz ** 2 * P_dz, dz)
                    plt.plot(time[list_dt_MSD], mean_dz_square*np.ones(NumberOfMSDPoint), label=r"$ \langle \Delta z \rangle $")

        plt.title(r"Mean square displacement on $" + axis +"$")
        plt.xlabel(r"$\tau$ $(\mathrm{s})$", fontsize=15)
        plt.ylabel(r"$\langle ($$" + axis + r"$$(t+\tau) - $$" + axis + r"$$(t))^2 \rangle$ $(\mathrm{m}^2)$", fontsize=15)
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.legend()
        plt.show()

    if output:
        return msd, time[list_dt_MSD]



def Cumulant4(axis, Rs, dt, Nt, plot=True, output=False):

    #--- def some array
    t = np.arange(0, Nt, dt)

    if axis == "x":
        position = Rs[:, 0]
    elif axis == "y":
        position = Rs[:, 1]
    elif axis == "z":
        position = Rs[:, 2]
    else:
        raise ValueError("WRONG AXIS : choose between 'x', 'y' and 'z' !")

    list_dt_c4 = np.array([], dtype=int)
    for i in range(len(str(Nt)) - 3):
        # Take just 10 points by decade.
        list_dt_c4 = np.concatenate(
            (
                list_dt_c4,
                np.arange(10 ** i, 10 ** (i + 1), 10 ** i, dtype=int),
            )
        )
    c4 = np.zeros(len(list_dt_c4))

    #--- Compute cumulant4
    for k, i in enumerate(tqdm(list_dt_c4)):
        if i == 0:
            c4[k] = 0
            continue
        c4[k] = (np.mean((position[i:] - position[:-i]) ** 4) - 3 * (
            np.mean((position[i:] - position[:-i]) ** 2)) ** 2) * 1 / (24)


    if plot:
        plt.loglog(t[list_dt_c4], c4, "o")

    if output:
        return t[list_dt_c4], c4


def PDF(axis, Rs, dt, Nt, D=None, lD=None, lB=None, N_tau=None, speed_drift=None, space=None, bins=50, plot=True, output=False):
    """

    Parameters
    ----------
    axis : "x", "y" or "z"
    Rs : datas to analyse
    dt : time step
    D : diffusion coeficient
    lD : Debye Lenght
    lB : Boltzman lenght
    speed_drift : value of drift speed
    space : "wall" or "bulk" domaine
    bins : number of chanel to compute PDF [default = 50]
    plot : True if plot is needed
    output : if True : return {z_hist, hist, pdf_theo}

    -------

    """
    if space!="bulk" and space!="wall":
        raise ValueError("WRONG SPACE : choose between 'bulk' and 'wall' !")
    # if axis!="x" and space!="y" and space!="z" and space!="dx" and space!="dy" and space!="dz":
    #     raise ValueError("WRONG AXIS : choose between 'x', 'y', 'z', 'dx', 'dy' and 'dz' !")
    # if len(Rs[:,0])!=Nt:
    #     raise ValueError("WRONG LENGHT : Nt need to be same lenght than Rs[i,:], for all i!")

    time = np.arange(0, Nt) * dt
    if N_tau==0:
        N_tau = 10

    # ------ What do you want ? ----
    if axis == "x":
        position = Rs[:, 0]

    elif axis == "y":
        position = Rs[:, 1]

    elif axis == "z":
        position = Rs[:, 2]

    elif axis == "dx":
        axis = "\Delta x"
        position = Rs[:, 0]
        dX = position[N_tau:] - position[:-N_tau]

    elif axis == "dy":
        axis = "\Delta y"
        position = Rs[:, 1]
        dX = position[N_tau:] - position[:-N_tau]

    elif axis == "dz":
        axis = "\Delta z"
        position = Rs[:, 2]
        dX = position[N_tau:] - position[:-N_tau]
    else:
        raise ValueError("WRONG AXIS : choose between 'x', 'y' and 'z' or 'dx', 'dy' and 'dz' !")

    # ---- Where do you want ? ----
    if space=="bulk":
        hist, bin_edges = np.histogram(dX, bins=bins, density=True)
        binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
        binsPositions = binsPositions / np.sqrt(2 * D * dt * N_tau)
        pdf = hist / np.trapz(hist, binsPositions)

        if speed_drift!=None:
            z_theo = np.linspace(-5, 5, 2000)
            PDFtheo = 1/np.sqrt(2*np.pi) * np.exp(-(z_theo)**2 / 2)
            z_theo = z_theo + speed_drift*N_tau*dt/np.sqrt(2 * D * dt * N_tau)
        else:
            z_theo = np.linspace(-5, 5, bins)
            PDFtheo = 1/np.sqrt(2*np.pi) * np.exp(-z_theo**2 / 2)


    if space=="wall":
        if axis=="x" or axis=="y" or axis=="z":
            hist, bin_edges = np.histogram(position[position < 3e-6], bins=bins, density=False)
            binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            zth = np.linspace(-5, 5, bins)
            PDF_gauss = 1/np.sqrt(2*np.pi) * np.exp(-(zth)**2 / 2) #Gaussian theory

            hist, bin_edges = np.histogram(dX, bins=bins, density=False)
            binsPositions = (bin_edges[:-1] + bin_edges[1:]) / 2
            binsPositions = binsPositions / (np.sqrt(2* D * dt * N_tau))
            Label = r"$\tau = "+str(N_tau*dt)+"\mathrm{s}$"

        pdf = hist / np.trapz(hist, binsPositions)

        if axis=="z" and lB!=None and lD!=None:
            z_theo = np.linspace(1e-9, 3e-6, bins)
            p_theo = np.exp(- 4.8 * np.exp(-z_theo / lD) - z_theo / lB)
            PDFtheo = p_theo / np.trapz(p_theo, z_theo)

        # if (axis=="dx" or axis=="dy" or axis=="dz") and lB!=None and lD!=None :
            # z_theo = np.linspace(1e-9, 3e-6, bins)
            # P_Di = np.trapz(D*P_z_wall(z_theo, 1, 4.8, lD, lB)), z_theo)
            # dxi_theo = np.linspace(-5, 5, bins)
            # P_dxi = np.trapz(P_Di*np.exp(-dxi_theo**2/(4*D*)))


    """
    ---------------- PLOTS
    """
    if plot:
        if space=="bulk":
            plt.semilogy(binsPositions,pdf, "o", markersize=4, label=r"$\mathrm{Numerical}$")
            plt.plot(z_theo, PDFtheo, "k-", label=r"$\mathrm{Theoritical}$" )

            plt.xlabel(r"$" + axis + "/ \sqrt{2D \Delta t} $")
            plt.ylabel(r"$P(" + axis + ") $")

        if space == "wall":
            plt.semilogy(binsPositions, pdf, "o", label=r"$\mathrm{Numerical}$")
            if axis=="z": # or axis=="\Delta x" or axis=="\Delta y" or axis=="\Delta z"
                plt.plot(z_theo, PDFtheo, "k-", label=r"$\mathrm{Theoritical}$")
                if axis!="z":
                    plt.plot(zth, PDF_gauss, "r:", label=r"$\mathrm{Gaussienne}$")
                    plt.title(Label)

            if axis=="x" or axis=="x" or axis=="z":
                plt.xlabel(r"$" + axis + "(\mathrm{m})$", fontsize=15)
                plt.ylabel(r"$P(" + axis + ") ~ (\mathrm{m}^{-1})$", fontsize=15)
            else:
                plt.xlabel(r"$" + axis + "/\sigma$", fontsize=15)
                plt.ylabel(r"$P(" + axis + ")$", fontsize=15)

        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend()
        plt.show()

    if output==True:
        return binsPositions, hist




def D_num(axis, Rs, dt, D, lD=None, lB=None, speed_drift=None, space=None, plot=True, output=False):
    """

    Parameters
    ----------
    axis : "x", "y" or "z"
    Rs : datas to analyse
    dt : time step
    D : diffusion coeficient
    lD : Debye Lenght
    lB : Boltzman lenght
    speed_drift : value of drift speed
    space : "wall" or "bulk" domaine
    plot : True if plot is needed
    output : if True : return MSD_axis, time

    -------

    """
    Nt, Np, dim = np.shape(Rs)
    time = np.arange(0, Nt) * dt

    puissance_dt = len(str(dt)) - 2
    puissance_Nt = len(str(Nt)) - 1

    if axis == "x":
        position = Rs[:, :, 0]
    elif axis == "y":
        position = Rs[:, :, 1]
    elif axis == "z":
        position = Rs[:, :, 2]
    else:
        raise ValueError("WRONG AXIS : choose between 'x', 'y' and 'z' !")

    list_n = np.array([], dtype=int)
    for i in range(puissance_Nt):
        # Take just 9 points by decade.
        list_n = np.concatenate(
            (
                list_n,
                np.arange(10 ** i, 10 ** (i + 1), 10 ** (i), dtype=int),
            )
        )
    NumberOfPoint = len(list_n)
    D = np.zeros(NumberOfPoint)
    for k, i in enumerate(tqdm(list_n)):
        if k == 0:
            D[k] = 0
            continue
        for n in range(Np):
            delta_x = position[i:, n]-position[0:-i, n]
            D[k] += np.std(delta_x)**2/(2*k*dt)
        D[k] = D[k]/Np


    if plot:
        plt.plot(time[list_n[:NumberOfPoint-49]], D[:NumberOfPoint-49], "o", label="Numerical")

        plt.title("Diffusion on " + axis)
        plt.xlabel(r"$\Delta t$ $[s]$", fontsize=20)
        plt.ylabel(r"$D_{"+ axis + "} [m²/s]$ ", fontsize=20) #
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.legend()
        plt.show()

#------------------------------------------------------------------------------------------------------------

def P_z_wall(z, A, B, lD, lB):
    # PDF(z)
    if type(z) != np.ndarray:
        if z < 0:
            return 0
        return A * np.exp(-(B) * np.exp(- z / lD) - z / lB)

    P = A * np.exp(-(B) * np.exp(- z / lD) - z / lB)
    P[z < 0] = 0

    return P


def Pz_PDeltaz(z, deltaz, A, B, lD, lB):
    # P(z)*P(z+Deltaz)
    PP = P_z_wall(z, A, B, lD, lB) * P_z_wall(z + deltaz, A, B, lD, lB)

    return PP


def _P_Deltaz_longTime(z, deltaz, A, B, lD, lB):
    # integrate of P(z)*P(z+Deltaz) on z
    PPP = np.trapz(Pz_PDeltaz(z, deltaz, A, B, lD, lB), z)

    return PPP


def P_Deltaz_longTime(deltaz, B, lD, lB):
    A = 1
    z = np.linspace(0, 1e-5, 1000)
    PPPP = np.zeros(len(deltaz))
    for i in range(len(deltaz)):
        PPPP[i] = _P_Deltaz_longTime(z, deltaz[i], A, B, lD, lB)
    A = 1 / np.trapz(PPPP, deltaz)

    return PPPP * A

######################################################



