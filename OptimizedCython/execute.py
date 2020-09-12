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

# cy = timeit.timeit(
#     "RigidWallOverdampedLangevin3D_cython.test()",
#     setup="import RigidWallOverdampedLangevin3D_cython",
#     number=1,
# )
# py = timeit.timeit(
#     "RigidWallOverdampedLangevin3D.test()", setup="import RigidWallOverdampedLangevin3D", number=1
# )

# """
#  ---------------------------------------
#  ------- Rigid wall INERTIAL ---------
#  ---------------------------------------
# """

cy = timeit.timeit(
    "RigidWall_InertialLangevin3D_cython.test()",
    setup="import RigidWall_InertialLangevin3D_cython",
    number=1,
)
py = timeit.timeit(
    "RigidWall_InertialLangevin3D.test()", setup="import RigidWall_InertialLangevin3D", number=1
)

"""
========================================================================
           SOME PRINTS
========================================================================
"""


print("----- Cython speed = {:.5f} s".format(cy))
print("----- Python speed = {:.5f} s".format(py))
print("-->Cython is {:.5f} x faster than Python !".format((py / cy)))


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