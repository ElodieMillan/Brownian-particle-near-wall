from testGSL import a
import time

if __name__ == "__main__":
    t = time.time()

    a()

    print(time.time() - t)

    import numpy as np

    dt = 1e-6
    Nt = 1000000
    t = time.time()
    rngx = (1 / np.sqrt(dt)) * np.random.default_rng().normal(0.0, 1, size=Nt)
    print(time.time() - t )