import time
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * x - 5 * x + 5


def df(x):
    return 2 * x - 5


N = 20
xx = 0

# lmd = 0.1
# lmd = 0.9
# lmd = 0.01
lmd = 0.3

x_plt = np.arange(0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

plt.ion()
fig, ax = plt.subplots()
ax.grid(True)

ax.plot(x_plt, f_plt)
point = ax.scatter(xx, f(xx), c='red')

mn = 100

for i in range(N):
    lmd = 1 / min(i + 1, mn)

    xx = xx - lmd * np.sign(df(xx))
    point.set_offsets([xx, f(xx)])

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)

plt.ioff()
print(xx)
ax.scatter(xx, f(xx), c='blue')
plt.show()
