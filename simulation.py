import numpy as np
import matplotlib.pyplot as plt


def get_p_and_y(n, x, alpha, b):
    """
    Generate n-vector of beta-distributed(alpha, b) probabilities and
    n-vector of observations
    """
    p = np.random.beta(alpha, b, size=n) # Vector containing all probabilities p_i
    # p=np.ones(n) * 0.51
    observed_correct = np.random.binomial(n=1, p=p, size=n) # Boolean vector indicating whether or not individual i
                                                            # observed correct
    # print("Observed correct: {}".format(observed_correct))
    if x == 1:
        return p, observed_correct
    else:
        return p, 1 - observed_correct

def create_a_i(z, p, a_vec_0, a_vec_1, i):
    """
    :param z: vector of previous observations
    :param p: vector of all probailities
    :param a_vec_0: a_i(z, x=0)
    :param a_vec_1: a_i(z, x=1)
    :param i: Current index
    :return: Two vectors containing a_i(z,x=0) and a_i(z, x=1) with a new value at index i.
    """

    a_vec_0[i] = np.sum([p[i] * ((z[i] == 1).astype(np.int) + (((z[i] == 0).astype(np.int) - (z[i] == 1).astype(np.int)) *
                ((p[i] * np.prod(a_vec_0[0:i])) > ((1-p[i]) * np.prod(a_vec_1[0:i]))))), (1-p[i]) * ((z[i]==1).astype(np.int)
            + (((z[i] == 0).astype(np.int) - (z[i] == 1).astype(np.int)) * ((((1 - p[i]) * np.prod(a_vec_0[0:i])) > p[i] * np.prod(a_vec_1[0:i])))))])

    a_vec_1[i] =  np.sum([(1 - p[i]) * ((z[i] == 1).astype(np.int) + (((z[i]==0).astype(np.int)
                        - (z[i] == 1).astype(np.int)) * (((p[i] * np.prod(a_vec_0[0:i]))) > ((1-p[i])
                * np.prod(a_vec_1[0:i]))))), p[i] * ((z[i] == 1).astype(np.int) + (((z[i] == 0).astype(np.int) - (z[i] == 1).astype(np.int)) *
                         (((1 - p[i]) * np.prod(a_vec_0[0:i])) > p[i] * np.prod(a_vec_1[0:i]))))])

    if ((p[i] * np.prod(a_vec_0[0:i])) == ((1-p[i]) * np.prod(a_vec_1[0:i]))) or (((1 - p[i]) * np.prod(a_vec_0[0:i])) == p[i] * np.prod(a_vec_1[0:i])):
        print("a_i terms equal")

    return a_vec_0, a_vec_1


def z(n, alpha, beta):
    """ Calculate the decision for each of the n individuals"""

    # np.random.seed(1)
    # Assuming x has a 50/50 chance of 0 or 1
    x = np.random.binomial(n=1, p=0.5, size=None)
    print("x = {}\n".format(x))

    p, y = get_p_and_y(n, x, alpha, beta) # get probabilities and observations
    # print("p = {}".format(p))
    # print("y= {}".format(y))

    z = np.zeros(n, dtype=int)
    a_vec_0 = np.zeros(n)
    a_vec_1 = np.zeros(n)
    prob_0 = np.zeros(n) # Container for argmax-expression inserted x=0
    prob_1 = np.zeros(n) # Container for argmax-expression inserted x=1

    if (p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1) == p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0)):
        z[0] = np.random.randint(0, 2)

    z[0] = (p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1) < p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0))
    prob_0[0] = p[0] * (y[0] == 0) + (1 - p[0]) * (y[0] == 1)
    prob_1[0] = p[0] * (y[0] == 1) + (1 - p[0]) * (y[0] == 0)

    a_vec_0[0] = np.sum([((z[0]==0) * (p[0] > 0.5) + (z[0] == 1) * (p[0] < 0.5)) * p[0],
                       ((z[0]==1) * (p[0] >= 0.5) + (z[0] == 0) * (p[0] < 0.5)) * (1 - p[0])])

    a_vec_1[0] = np.sum([((z[0] == 0) * (p[0] > 0.5) + (z[0] == 1) * (p[0] < 0.5)) * (1-p[0]),
                       ((z[0]==1) * (p[0] >= 0.5) + (z[0] == 0) * (p[0] < 0.5)) * p[0]])

    for i in range(1, n):

        if ((p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * np.prod(a_vec_0[0:i])) == ((p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * np.prod(a_vec_1[0:i])):
            print("Equal")
            z[i] = np.random.randint(0,2)
        else:
            z[i] = (((p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * np.prod(a_vec_0[0:i])) <
                ((p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * np.prod(a_vec_1[0:i])))
        prob_0[i] = (p[i] * (y[i] == 0) + (1 - p[i]) * (y[i] == 1)) * np.prod(a_vec_0[0:i])
        prob_1[i] = (p[i] * (y[i] == 1) + (1 - p[i]) * (y[i] == 0)) * np.prod(a_vec_1[0:i])
        a_vec_0, a_vec_1 = create_a_i(z, p, a_vec_0, a_vec_1, i)


    print(a_vec_0)
    print(a_vec_1)
    return z, x, y, p, prob_0, prob_1




