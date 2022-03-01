import numpy as np
import matplotlib.pyplot as plt
from Cumulant4_fonction import C4_long, C4_court, Cross_time
from DoubleRigidWallOverdampedBrownExp_Cython import RigidWallOverdampedLangevin3D
from scipy.integrate import quad
from scipy.io import loadmat
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import trapz
from scipy import interpolate


def cm2inch(value):
    return value / 2.54

from tqdm import tqdm
# axes.xaxis.set_tick_params(direction='in', which='both')
# axes.yaxis.set_tick_params(direction='in', which='both')

import sys

dt = float(sys,argv[1])
Nt = int(sys.argv[2])
Nt_sub = int(sys.argv[3])
a = float(sys.argv[4])
lD = float(sys.argv[5])
H = float(sys.argv[6])
nb_simu = int(sys.argv[7])

filename = "simulation_C4_dt_{}_Nt_{}_Nsub_{}_lB_{}_lD_{}_H_{}.csv".format(*[dt, Nt, N_sub, lB, lD, H])


def compute(n):
    simu = RigidWallOverdampedLangevin3D(dt=dt, Nt=Nt, a=a, H=H, lD=lD, Nt_sub=1)
    simu.trajectory()
    tau, c4 = simu.Cumulant4("x", plot=False, output=True)
    csvfile = open(filename, 'a', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(c4)
    tau4_simu.append(tau)
    C4_simu.append(c4)


from multiprocessing import Pool

with Pool(multiprocessing.cpu_count()) as p:
    p.map(compute, range(nb_simu))