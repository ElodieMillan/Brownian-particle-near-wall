import timeit
import sys

sys.path.append(r"C:\Users\Shadow\PycharmProjects\StageObjet2020\PurePython")

"""
 ---------------------------------------
 ------- Overdamped without wall -------
 ---------------------------------------
"""

# cy = timeit.timeit(
#     "OverdampedLangevin3D_cython.test()",
#     setup="import OverdampedLangevin3D_cython",
#     number=1,
# )
# py = timeit.timeit(
#     "OverdampedLangevin3D.test()", setup="import OverdampedLangevin3D", number=1
# )

"""
 ---------------------------------------
 ------- Inertial without wall ---------
 ---------------------------------------
"""
#
# cy = timeit.timeit(
#     "InertialLangevin3D_cython.test()",
#     setup="import InertialLangevin3D_cython",
#     number=1,
# )
#
# py = timeit.timeit(
#     "InertialLangevin3D.test()", setup="import InertialLangevin3D", number=1
# )

"""
 ---------------------------------------
 ------- Rigid wall Overdamped ---------
 ---------------------------------------
"""

cy_RWO = timeit.timeit(
    "RigidWallOverdampedLangevin3D_cython.test()",
    setup="import RigidWallOverdampedLangevin3D_cython",
    number=1,
)
py_RWO = timeit.timeit(
    "RigidWallOverdampedLangevin3D.test()", setup="import RigidWallOverdampedLangevin3D", number=1
)

print("=========== RIGID WALL OVERDAMPED ============")
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

cy_RWI = timeit.timeit(
    "RigidWall_InertialLangevin3D_cython.test()",
    setup="import RigidWall_InertialLangevin3D_cython",
    number=1,
)
py_RWI = timeit.timeit(
    "RigidWall_InertialLangevin3D.test()", setup="import RigidWall_InertialLangevin3D", number=1
)

print("=========== RIGID WALL INERTIAL =============")
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

# from RigidWall_InertialLangevin3D_cython import RigidWallInertialLangevin3D
# langevin3D = RigidWallInertialLangevin3D(dt=1e-6, Nt=1000000, R=1.5e-6, rho=2500, x0=(0.0, 0.0, 1e-7))
# langevin3D.trajectory()
# # print("done")
# #print(langevin3D.x)
# langevin3D.plotTrajectory()