import time
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Ensure GUI backend is used
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Cost function
def E(y, a, b):
    ff = np.array([a * z + b for z in range(N)])
    return np.dot((y - ff).T, (y - ff))


# Gradients
def dEda(y, a, b):
    ff = np.array([a * z + b for z in range(N)])
    return -2 * np.dot((y - ff).T, range(N))


def dEdb(y, a, b):
    ff = np.array([a * z + b for z in range(N)])
    return -2 * (y - ff).sum()


# Parameters
N = 100
Niter = 50
sigma = 3
at = 0.5
bt = 2
aa = 0
bb = 0
lmd1 = 0.000001
lmd2 = 0.0005

# Data
f = np.array([at * z + bt for z in range(N)])
y = np.array(f + np.random.normal(0, sigma, N))

# Precompute E surface
a_plt = np.arange(-1, 2, 0.1)
b_plt = np.arange(0, 3, 0.1)
E_plt = np.array([[E(y, a, b) for a in a_plt] for b in b_plt])
a, b = np.meshgrid(a_plt, b_plt)

# Plot setup
plt.ion()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(a, b, E_plt, color='y', alpha=0.5)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('E')

# Initialize point
point = ax.scatter(aa, bb, E(y, aa, bb), c='red')
a_path, b_path, e_path = [], [], []

# Gradient descent loop
for n in range(Niter):
    aa -= lmd1 * dEda(y, aa, bb)
    bb -= lmd2 * dEdb(y, aa, bb)

    e_val = E(y, aa, bb)
    a_path.append(aa)
    b_path.append(bb)
    e_path.append(e_val)

    # Update point
    point.remove()
    point = ax.scatter(aa, bb, e_val, c='red')

    # Optionally draw path
    if n > 0:
        ax.plot(a_path[-2:], b_path[-2:], e_path[-2:], c='red', alpha=0.7)

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.05)  # Slightly longer pause to see clearly

    print(f"Iter {n + 1}: a = {aa:.5f}, b = {bb:.5f}, E = {e_val:.2f}")

plt.ioff()
plt.show()

ff = np.array([aa * z + bb for z in range(N)])

plt.scatter(range(N), y, s=2, c='red')
plt.plot(f)
plt.plot(ff, c='red')
plt.grid(True)
plt.show()
