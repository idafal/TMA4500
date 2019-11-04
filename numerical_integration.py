import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import simulation as sim
from scipy.stats import beta


# print(f(np.arange(-1,1.5,0.5)))
# x_range = np.arange(-5,5.02, 0.02)
# plt.plot(x_range,f(x_range))
# plt.show()
# integrated = integrate.quad(f, -5, 5)
# print(integrated)

n=2
alpha = 8.5
b = 3

x = np.random.binomial(n=1, p=0.5, size=None)
p, y = sim.get_p_and_y(n, x, alpha=alpha, b=b)

z = np.zeros(n)

z[0] = (p[0] * (y[0]==0) + (1-p[0]) * (y[0] == 1) < p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0))
print("real z0={}".format(z[0]))
f = lambda p1: beta.pdf(p1, alpha, b)

def I_x0y0(p1, z1):
    print("z1 = {}".format(z1))
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * (p1 > (1-p1))) + (z1 == 1)) * p1 * f(p1)
def I_x0y1(p1, z1):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * ((1-p1) > p1)) + (z1 == 1)) * (1-p1) * f(p1)
def I_x1y0(p1, z1):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * (p1 > (1 - p1))) + (z1 == 1)) * (1 - p1) * f(p1)
def I_x1y1(p1, z1):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * ((1 - p1) > p1)) + (z1 == 1)) * (1 - p1) * f(p1)


def z2(p2, z):

    x_0 = np.sum([p2 * integrate.quad(I_x0y0, 0, 1, args=(z[0]))[0], (1-p2) * integrate.quad(I_x0y1, 0, 1, args=(z[0]))[0]]) #[y=0, y=1]
    x_1 = np.sum([(1 - p2) * integrate.quad(I_x1y0, 0, 1, args=(z[0]))[0], p2 * integrate.quad(I_x1y1, 0,1, args=(z[0]))[0]])   #[y=0, y=1]

    return (x_0 < x_1)

z[1] = z2(p[1], z)
print("p: {}".format(p))
print("x = {}".format(x))
print("y: {}".format(y))
print("z: {}".format(z))