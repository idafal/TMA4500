import numpy as np
import plot_functions as plot
import simulation as sim
import matplotlib.pyplot as plt
import time

if __name__=='__main__':
    MasterFlag = {
        0: '0: Calculate z_i',
        1: '1: Calculate z_i and plot result',
        2: '2: Calculate z_i and plot result and probabilities',
    }[1]
    print(MasterFlag)
    if MasterFlag == '0: Calculate z_i':
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1 = sim.z(n=100000, alpha=6, beta=5)
        end = time.time()
        print("Time elsapsed = {}".format(end - start))
        print("Decision z = {}".format(z_vector))

    elif MasterFlag == '1: Calculate z_i and plot result':
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1 = sim.z(n=100, alpha=5, beta=5)
        end = time.time()
        print("Time elsapsed = {}".format(end - start))
        plot.decisions(x, z_vector, y, p, include_observations=True)

    elif MasterFlag == '2: Calculate z_i and plot result and probabilities':
        z_vector, x, y, p, prob_0, prob_1 = sim.z(n=100, alpha=6.5, beta=6)
        plot.decisions(x, z_vector, y, p, include_observations=True, include_probs=False)
        plot.argmax_sizes(prob_0, prob_1, p)
