import numpy as np
import scipy.integrate as integrate
from scipy.stats import beta
import simulation as sim

n=2
alpha = 5
b = 3

f_p = lambda pi: beta.pdf(pi, alpha=alpha, b=b)
f_p_tilde = lambda p_tilde_i: beta.pdf(p_tilde_i, alpha=alpha, b=b)
a_vec_0 = np.zeros(n)
a_vec_1 = np.zeros(n)

## Base-case integrands for a_i-recursion
def I_x0y0(p1, z1):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * (p1 > (1 - p1))) + (z1 == 1)) * p1 * f_p(p1)
def I_x0y1(p1, z1):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * ((1-p1) > p1)) + (z1 == 1)) * (1-p1) * f_p(p1)
def I_x1y0(p1, z1):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * (p1 > (1 - p1))) + (z1 == 1)) * (1 - p1) * f_p(p1)
def I_x1y1(p1, z1):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * ((1 - p1) > p1)) + (z1 == 1)) * (1 - p1) * f_p(p1)

def z(n, alpha, b):
    np.random.seed(1)
    x = np.random.binomial(n=1, p=0.5, size=None)
    p, y = sim.get_p_and_y(n, x, alpha, beta=b)
    # Container for decisions
    z = np.zeros(n, dtype=int)
    # Base-case
    z[0] = (p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1) < p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0))
    if (p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1) == p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0)):
        z[0] = np.random.randint(0, 2)

    a_vec_0 = np.zeros(n)
    a_vec_1 = np.zeros(n)
    a_vec_0[0] = np.sum([integrate.quad(I_x0y0, 0, 1, args=(z[0]))[0], integrate.quad(I_x0y1, 0, 1, args=(z[0]))[0]])
    a_vec_1[0] = np.sum([integrate.quad(I_x1y0, 0, 1, args=(z[0]))[0], integrate.quad(I_x1y1, 0, 1, args=(z[0]))[0]])

    for i in range(1,n):
        if ((p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * a_vec_1[i-1]) == ((p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * a_vec_1[i-1]):
            print("Equal")
            z[i] = np.random.randint(0,2)
        else:
            z[i] = (((p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * a_vec_0[i - 1]) <
                    ((p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * a_vec_1[i - 1]))
