import numpy as np
import matplotlib.pyplot as plt

N = 100
dNoise = 1
dSignal = 5
r = 0.99
en = 0.1

x = np.zeros(N)
x[0] = np.random.normal(0, dSignal)
for i in range(1, N):
    x[i] = r * x[i - 1] + np.random.normal(0, en)

z = x + np.random.normal(0, dNoise, N)

# Filter  Kalman for moving objects
xx = np.zeros(N)
P = np.zeros(N)
xx[0] = z[0]
P[0] = dNoise

for i in range(1,N):
    Pe = r*r*P[i-1] + en * en
    P[i] = (Pe*dNoise) / (Pe+dNoise)
    xx[i] = r*xx[i-1] + P[i]/dNoise*(z[i] - r*xx[i-1])

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols =1, figsize=(10,5))

ax1.plot(x)
ax1.plot(z)
ax1.plot(xx)
ax1.grid(True)

ax2.plot(P)
ax2.grid(True)
plt.show()





