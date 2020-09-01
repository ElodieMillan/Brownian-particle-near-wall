import timeit
import sys

sys.path.append(r"C:\Users\Shadow\PycharmProjects\StageObjet2020\PurePython")

# ---------------------------------------
# ------- Overdamped without wall -------
# ---------------------------------------

# cy = timeit.timeit(
#     "OverdampedLangevin3D_cython.test()",
#     setup="import OverdampedLangevin3D_cython",
#     number=1,
# )
# py = timeit.timeit(
#     "OverdampedLangevin3D.test()", setup="import OverdampedLangevin3D", number=1
# )

# ---------------------------------------
# ------- Inertial without wall ---------
# ---------------------------------------

# cy = timeit.timeit(
#     "InertialLangevin3D_cython.test()",
#     setup="import InertialLangevin3D_cython",
#     number=1,
# )
# py = timeit.timeit(
#     "InertialLangevin3D.test()", setup="import InertialLangevin3D", number=1
# )

# ---------------------------------------
# ------- Rigid wall Overdamped ---------
# ---------------------------------------

cy = timeit.timeit(
    "RigidWallOverdampedLangevin3D_cython.test()",
    setup="import RigidWallOverdampedLangevin3D_cython",
    number=1,
)
py = timeit.timeit(
    "RigidWallOverdampedLangevin3D.test()", setup="RigidWallOverdampedLangevin3D", number=1
)

# ---------------------------------------
# -------------- Some print -------------
# ---------------------------------------


print("----- Cython speed = {:.5f} s".format(cy))
print("----- Python speed = {:.5f} s".format(py))
print("-->Cython is {:.5f} x faster than Python !".format((py / cy)))
