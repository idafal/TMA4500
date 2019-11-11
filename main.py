import numpy as np
import plot_functions as plot
import matplotlib.pyplot as plt
import time
import scipy.integrate as integrate

import simulation as sim
import simulation_model2 as model2

if __name__=='__main__':
    MasterFlag = {
        0: '0: Calculate z_i',
        1: '1: Calculate z_i and plot result',
        2: '2: Calculate z_i and plot result and probabilities',
        3: '3: Calculate z_i for model 2',
        4: '4: Test',
    }[2]
    print(MasterFlag)
    if MasterFlag == '0: Calculate z_i':
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1 = sim.z(n=100000, alpha=6, beta=5)
        end = time.time()
        print("Time elsapsed = {}".format(end - start))
        print("Decision z = {}".format(z_vector))

    elif MasterFlag == '1: Calculate z_i and plot result':
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1 = sim.z(n=300, alpha=1000, beta=1000)
        end = time.time()
        print("Time elsapsed = {}".format(end - start))
        plot.decisions(x, z_vector, y, p, include_observations=True)

    elif MasterFlag == '2: Calculate z_i and plot result and probabilities':
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1 = sim.z(n=300, alpha=6.5, beta=6)
        end = time.time()
        print("Time elsapsed = {}".format(end - start))
        plot.decisions(x, z_vector, y, p, include_observations=True, include_probs=True)
        plot.argmax_sizes(prob_0, prob_1, p)

    elif MasterFlag == '3: Calculate z_i for model 2':
        start=time.time()
        z_vector, x, y, p, a_vec_0, a_vec_1 = model2.z_i(n=50, alpha=50, b=40)
        end=time.time()
        print("Time elapsed = {}".format(end-start))
        plot.decisions(x, z_vector, y, p, include_observations=True, include_probs=True, grid=False)

    elif MasterFlag == '4: Test':
        delta_p = np.linspace(0, 1, 1000)
        f_0 = np.zeros_like(delta_p)
        f_1 = np.zeros_like(delta_p)
        g_0 = np.zeros_like(delta_p)
        g_1 = np.zeros_like(delta_p)

        zi = np.int64(1)
        z_vector, x, y, p, a_vec_0, a_vec_1 = model2.z_i(n=2, alpha=10, b=4)
        for j in range(0, len(delta_p)):
            f_0[j]=model2.I_x0y0_i(delta_p[j], zi, a_vec_0, 2, alpha=10, b=4)
            f_1[j] = model2.I_x0y1_i(delta_p[j], zi, a_vec_1, 2, alpha=10, b=4)
            g_0[j] = model2.I_x1y0_i(delta_p[j], zi, a_vec_0, 2, alpha=10, b=4)
            g_1[j] = model2.I_x1y1_i(delta_p[j], zi, a_vec_1, 2, alpha=10, b=4)

        plt.plot(delta_p, f_0, label="x=0, y=0")
        plt.plot(delta_p, f_1, label="x=0, y=1")
        plt.plot(delta_p, g_0, label="x=1, y=0")
        plt.plot(delta_p, g_1, label="x=1, y=1")
        plt.legend()
        plt.show()
        non_zero_indices = np.argwhere(f_1 != 0).flatten()
        int_true = integrate.quad(model2.I_x1y1_i, 0,1, args=(zi, a_vec_0, 2, 10, 4))
        int_log = integrate.quad(model2.I_x1y1_i, 0.5,1, args=(zi, a_vec_0, 2, 10, 4,True))
        print(int_true[0])
        print((int_log[0]))
        # print(non_zero_indices)
        # f_1 = np.log(f_1[non_zero_indices])
        # plt.plot(f_1)
        # plt.show()