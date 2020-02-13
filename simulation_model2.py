import numpy as np
import scipy.integrate as integrate
from scipy.stats import beta
import simulation as sim

def f_pi(pi, alpha, b):
    return beta.pdf(pi, a=alpha, b=b)

def z_i(n, alpha, b):
    x = np.random.binomial(n=1, p=0.5, size=None)
    x=0
    p, y = sim.get_p_and_y(n, x, alpha, b)
    # y[49] = 1
    # p[49] = 1
    # y[0] = 1-x
    # y[1] = 1-x

    z = np.zeros(n, dtype=int)     # Container for decisions
    prob_0 = np.zeros(n) # Container for argmax-expression inserted x=0
    prob_1 = np.zeros(n) # Container for argmax-expression inserted x=1
    print(p[0])
    # Base-case
    z[0] = (p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1) < p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0))
    if (p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1) == p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0)):
        z[0] = np.random.randint(0, 2)
    prob_0[0] = p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1)
    prob_1[0] = p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0)

    a_vec_0 = np.zeros(n)
    a_vec_1 = np.zeros(n)
    b_vec_0 = np.zeros(n)
    b_vec_1 = np.zeros(n)
    a_vec_0[0] = np.sum([integrate.quad(I_x0y0, 0, 1, args=(z[0], alpha, b))[0], integrate.quad(I_x0y1, 0, 1, args=(z[0], alpha, b))[0]])
    a_vec_1[0] = np.sum([integrate.quad(I_x1y0, 0, 1, args=(z[0], alpha, b))[0], integrate.quad(I_x1y1, 0, 1, args=(z[0], alpha, b))[0]])
    b_vec_0[0] = a_vec_0[0]
    b_vec_1[0] = a_vec_1[0]

    for i in range(1,n):
        if ((p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * a_vec_1[i-1]) == ((p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * a_vec_1[i-1]):
            print("Equal")
            print(a_vec_1[i-1])
            z[i] = np.random.randint(0,2)
        else:
            z[i] = (((p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * a_vec_0[i - 1]) <
                    ((p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * a_vec_1[i - 1]))

        prob_0[i] = (p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * a_vec_0[i - 1]
        prob_1[i] = (p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * a_vec_1[i - 1]

        a_vec_0, a_vec_1, b_vec_0[i], b_vec_1[i] = create_a_i(z, a_vec_0, a_vec_1, i, alpha, b)

    return z, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1

## Base-case integrands for a_i-recursion
def I_x0y0(p1, z1, alpha, b):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * (p1 > (1 - p1))) + (z1 == 1)) * p1 * f_pi(p1, alpha, b)

def I_x0y1(p1, z1, alpha, b):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * ((1-p1) > p1)) + (z1 == 1)) * (1-p1) * f_pi(p1, alpha, b)

def I_x1y0(p1, z1, alpha, b):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * (p1 > (1 - p1))) + (z1 == 1)) * (1 - p1) * f_pi(p1, alpha, b)

def I_x1y1(p1, z1, alpha, b):
    return ((((z1 == 0).astype(np.int) - (z1 == 1).astype(np.int)) * ((1 - p1) > p1)) + (z1 == 1)) * p1 * f_pi(p1, alpha, b)

# recursion integrands for iteration i
def I_x0y0_i(pi, zi, a_vec_0, a_vec_1, i, alpha, b):
    return ((((zi == 0).astype(np.int) - (zi == 1).astype(np.int)) * (pi * a_vec_0[i-1] > (1 - pi) * a_vec_1[i-1])) +
            (zi == 1)) * pi * f_pi(pi, alpha, b)

def I_x0y1_i(pi, zi, a_vec_0, a_vec_1, i, alpha, b):
    return ((((zi == 0).astype(np.int) - (zi == 1).astype(np.int)) * ((1 - pi) * a_vec_0[i-1] > pi * a_vec_1[i-1])) +
            (zi == 1)) * (1 - pi) * f_pi(pi, alpha, b)

def I_x1y0_i(pi, zi, a_vec_0, a_vec_1, i, alpha, b):
    return ((((zi == 0).astype(np.int) - (zi == 1).astype(np.int)) * (pi * a_vec_0[i-1] > (1 - pi) * a_vec_1[i-1])) + (zi == 1)) * (1 - pi) * f_pi(pi, alpha, b)

def I_x1y1_i(pi, zi, a_vec_0, a_vec_1, i, alpha, b):
    return ((((zi == 0).astype(np.int) - (zi == 1).astype(np.int)) * ((1 - pi) * a_vec_0[i-1] > pi * a_vec_1[i-1]))
                    + (zi == 1)) * pi * f_pi(pi, alpha, b)


def create_a_i(z, a_vec_0, a_vec_1, i, alpha, b):
    """
    :param a_vec_0:
    :param a_vec_1:
    :return:
    """
    b_0 = np.sum([integrate.quad(I_x0y0_i, 0, 1, args=(z[i], a_vec_0, a_vec_1, i, alpha, b))[0], integrate.quad(I_x0y1_i, 0, 1, args=(z[i],
            a_vec_0, a_vec_1, i, alpha, b))[0]])
    b_1 = np.sum([integrate.quad(I_x1y0_i, 0, 1, args=(z[i],a_vec_0, a_vec_1, i, alpha, b))[0],integrate.quad(I_x1y1_i, 0, 1, args=(z[i],
            a_vec_0, a_vec_1, i, alpha, b))[0]])

    a_vec_0[i] = b_0 * a_vec_0[i-1]
    a_vec_1[i] = b_1 * a_vec_1[i-1]



    return a_vec_0, a_vec_1, b_0, b_1
