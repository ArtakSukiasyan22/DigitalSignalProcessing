import numpy as np
import matplotlib.pyplot as plt

N = 30
dNoise = 0.01
en = 0.01
a = 0.9
x0 = 76
ex = en/(1-a*a)

x = np.zeros(N)
x[0] = 0
for i in range(1, N):
    x[i] = a*x[i-1] + np.random.normal(0, en)

x += x0

z = x + np.random.normal(0, dNoise, N)

R = np.array([[a**np.abs(i-j) for j in range(N)] for i in range(N)])
V = np.eye(N)*dNoise
RVinv = np.linalg.inv(R+V/ex)

mz = z.mean()
xx = np.zeros(N)
for k in range(N):
    alfa = np.dot(R[:, k], RVinv)
    xx[k] = np.dot(alfa, (z-mz)) + mz

fig, ax = plt.subplots()
ax.plot(x)
ax.plot(xx)
ax.plot(z)

ax.grid(True)
plt.show()