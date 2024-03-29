import timeit
import sys

sys.path.append(r"../Code-Pure-Python")
nb_run = 10

"""
 ---------------------------------------
 ------- Overdamped without wall -------
 ---------------------------------------
"""
print("================= OVERDAMPED ================")
print(" ---------- Nombre de run : {:d} ------------".format(nb_run))

cy_O = timeit.timeit(
    "OverdampedLangevin3D_cython.test()",
    setup="import OverdampedLangevin3D_cython",
    number=nb_run,
)/nb_run
py_O = timeit.timeit(
    "OverdampedLangevin3D.test()",
    setup="import OverdampedLangevin3D",
    number=nb_run
)/nb_run

print("----- Cython speed = {:.5f} s".format(cy_O))
print("----- Python speed = {:.5f} s".format(py_O))
print("-->Cython is {:.5f} x faster than Python !".format((py_O / cy_O)))
print("=============================================")
print("")

"""
 ---------------------------------------
 ------- Inertial without wall ---------
 ---------------------------------------
"""
print("================= INERTIAL ==================")
print(" ---------- Nombre de run : {:d} ------------".format(nb_run))

cy_I = timeit.timeit(
    "InertialLangevin3D_cython.test()",
    setup="import InertialLangevin3D_cython",
    number=nb_run,
)/nb_run

py_I = timeit.timeit(
    "InertialLangevin3D.test()",
    setup="import InertialLangevin3D",
    number=nb_run
)/nb_run

print("----- Cython speed = {:.5f} s".format(cy_I))
print("----- Python speed = {:.5f} s".format(py_I))
print("-->Cython is {:.5f} x faster than Python !".format((py_I / cy_I)))
print("=============================================")
print("")

"""
 ---------------------------------------
 ------- Rigid wall Overdamped ---------
 ---------------------------------------
"""

print("=========== RIGID WALL OVERDAMPED ============")
print(" ---------- Nombre de run : {:d} ------------".format(nb_run))

cy_RWO = timeit.timeit(
    "RigidWallOverdampedLangevin3D_cython.test()",
    setup="import RigidWallOverdampedLangevin3D_cython",
    number=nb_run,
)/nb_run
py_RWO = timeit.timeit(
    "RigidWallOverdampedLangevin3D.test()",
    setup="import RigidWallOverdampedLangevin3D",
    number=nb_run
)/nb_run

print("----- Cython speed = {:.5f} s".format(cy_RWO))
print("----- Python speed = {:.5f} s".format(py_RWO))
print("-->Cython is {:.5f} x faster than Python !".format((py_RWO / cy_RWO)))
print("=============================================")
print("")

"""
 ---------------------------------------
 ------- Rigid wall INERTIAL ---------
 ---------------------------------------
"""
print("=========== RIGID WALL INERTIAL =============")
print(" ---------- Nombre de run : {:d} ------------".format(nb_run))

cy_RWI = timeit.timeit(
    "RigidWall_InertialLangevin3D_cython.test()",
    setup="import RigidWall_InertialLangevin3D_cython",
    number=nb_run,
)/nb_run
py_RWI = timeit.timeit(
    "RigidWall_InertialLangevin3D.test()",
    setup="import RigidWall_InertialLangevin3D",
    number=nb_run,
)/nb_run

print("----- Cython speed = {:.5f} s".format(cy_RWI))
print("----- Python speed = {:.5f} s".format(py_RWI))
print("-->Cython is {:.5f} x faster than Python !".format((py_RWI / cy_RWI)))
print("=============================================")


""" 
========================================================================
           TESTS PRINTS 
========================================================================
"""


# from RigidWallOverdampedLangevin3D_cython import RigidWallOverdampedLangevin3D
# langevin3D = RigidWallOverdampedLangevin3D(dt=1e-3, Nt=1000000, R=1.5e-6, rho=1050, x0=(0., 0., 1.5e-6))
# langevin3D.trajectory()
# langevin3D.plotTrajectory()
#
# from RigidWall_InertialLangevin3D_cython import RigidWallInertialLangevin3D
# langevin3D = RigidWallInertialLangevin3D(dt=1e-6, Nt=1000000, R=1.5e-6, rho=2500, x0=(0.0, 0.0, 1e-7))
# langevin3D.trajectory()
# # print("done")
# #print(langevin3D.x)
# langevin3D.plotTrajectory()

""" 
========================================================================
           PROFILE.PY
========================================================================
"""

#import pstats, cProfile

#import pyximport
#pyximport.install()

#import RigidWall_InertialLangevin3D_cython

#cProfile.runctx("RigidWall_InertialLangevin3D_cython.test()", globals(), locals(), "Profile.prof")

#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()
