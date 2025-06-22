import numpy as np


def getL(y1, m1_F, m1_M, d1):
    res = 1 / (2 * d1) * ((y1 - m1_F) ** 2 - (y1 - m1_M) ** 2)
    return res


def getL0(p0, p1):
    return np.log(p0 / p1)


m1_F = 60
m1_M = 85

d1 = 9

p0 = 0.48
p1 = 0.52

N = 100

L0 = getL0(p0, p1)

nM = 0
for i in range(N):
    y1 = np.random.normal(m1_F, d1)

    L = getL(y1, m1_F, m1_M, d1)

    if (L >= L0):
        print("Male", y1)
        nM += 1
    else:
        print("Female", y1)

error = nM / N * 100
print(f"Error: {np.round(error, 2)}%")