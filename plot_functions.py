import numpy as np
import matplotlib.pyplot as plt

def decisions(x, z_vector, observations,p, include_observations=False, include_probs=False, block=True, grid=False):
    plt.plot(z_vector)
    plt.plot(z_vector, 'bo--', label="Decision (z)")
    if include_observations:
        plt.plot(observations, color='grey', marker='o', linestyle='dashed', label="Observation (y)", alpha=0.5)
    plt.xlabel("Individual")
    if include_probs:
        plt.plot(p, label=r"Probability of observing correct $P(Y=x|X=x)$", alpha=0.7)
    plt.legend()
    plt.grid(grid)
    plt.title("True value: x = {}".format(x))
    plt.show(block=block)

def probabilities(p, block=True):
    plt.plot(p)
    plt.xlabel("Individual")
    plt.ylabel('probability')
    plt.legend()
    plt.title("Probability of observing correct")
    plt.show(block=False)
    plt.show()

def argmax_sizes(prob_0, prob_1, p):
    plt.plot(prob_0, label="x=0")
    plt.plot(prob_1, label="x=1")
    # plt.plot(p, label = "$p_i$")
    plt.ylabel("Probability")
    plt.xlabel("Individual")
    plt.legend()
    plt.title(r"$\argmax_x$P(X=x|$\bf{Y}$, $\bf{p}$) ")
    plt.show()