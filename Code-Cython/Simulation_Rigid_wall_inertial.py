import numpy as np
import matplotlib.pyplot as plt
import pyximport
import h5py
from tqdm import tqdm

pyximport.install(setup_args={'include_dirs': np.get_include()},reload_support=True)
from RigidWall_InertialLangevin3D_cython import RigidWallInertialLangevin3D

from mpl_toolkits import mplot3d
from scipy.signal import correlate

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 140

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["lines.markeredgecolor"] = "k"
mpl.rcParams["lines.markeredgewidth"] = 0.1
mpl.rcParams["figure.dpi"] = 130
from matplotlib import rc
rc('font', family='serif')
#rc('text', usetex=True)
rc('xtick', labelsize='x-small')
rc('ytick', labelsize='x-small')


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """

    def __init__(self, datapath, dataset, shape, dtype=np.float64, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape)

    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1,) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()



langevin3D = RigidWallInertialLangevin3D(dt=1e-7, Nt=10000000, R=1.5e-6, rho=1050, x0=(0., 0., 150e-9))
langevin3D.trajectory()

store_full = HDF5Store("Datas_inertial_RIgidWall\inertial_simulation_rigid_rigid_wall_r1p5em6_rho1050_B4_ld_70em9_dt1em7full_dt1em2reduced_10kpoints.h5", "full_data_1run", shape = (3, 10000000))

store_full.append([langevin3D.x, langevin3D.y, langevin3D.z])

store_reduced = HDF5Store("Datas_inertial_RIgidWall\inertial_simulation_rigid_rigid_wall_r1p5em6_rho1050_B4_ld_70em9_dt1em7full_dt1em2reduced_10kpoints.h5", "reduced_data_1run", shape = (3,9999990))

tmp = np.zeros((3,9999990))

for i in tqdm(range(10010)):
    langevin3D.x0 = np.array([langevin3D.x[-2:], langevin3D.y[-2:], langevin3D.z[-2:]])
    langevin3D.trajectory()
    tmp[:, i:i + 999] = np.array((langevin3D.x, langevin3D.y, langevin3D.z))[:, ::10000][:, 1:]

store_reduced.append(tmp)

print("Thanks for using me !")

