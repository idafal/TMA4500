import numpy as np
import plot_functions as plot
import matplotlib.pyplot as plt
import time
import scipy.integrate as integrate
from scipy.stats import beta

import simulation as sim
import simulation_model2 as model2

# np.random.seed(713)

if __name__=='__main__':
    MasterFlag = {
        0: '0: Calculate z_i',
        1: '1: Calculate z_i and plot result',
        2: '2: Calculate z_i and plot result and probabilities',
        3: '3: Calculate z_i for model 2',
        4: '4: Model 1: How fast does a cascade occur?',
        5: '5: Model 2: How fast does a cascade occur?',
        6: '6: How fast does a cascade occur?',
        7: '7: Model 1: Is the cascade correct?',
        8: '8: Model 2: Is the cascade correct?',
        9: '9: Is the cascade correct?',
        10: '10: Model 1: Is the cascade stable?',
        11: '11: Model 2: Is the cascade stable?',
        12: '12: Is the cascade stable?',
        13: '13: Plots',
        14: '14: Test',
    }[13]
    print(MasterFlag)
    if MasterFlag == '0: Calculate z_i':
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1 = sim.z(n=10, alpha=6, beta=5)
        end = time.time()
        print(b_vec_0, b_vec_1)
        print("Time elsapsed = {}".format(end - start))
        print("Decision z = {}".format(z_vector))

    elif MasterFlag == '1: Calculate z_i and plot result':
        alpha = 20
        b = 9
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1 = sim.z(n=100, alpha=alpha, beta=b)
        end = time.time()
        print("Time elsapsed = {}".format(end - start))
        plot.decisions(x, z_vector, y, p, alpha, b, include_observations=False, include_probs=True)

    elif MasterFlag == '2: Calculate z_i and plot result and probabilities':
        alpha = 15
        b = 6
        start = time.time()
        z_vector, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1 = sim.z(n=25, alpha=alpha, beta=b)
        end = time.time()
        print("Time elsapsed = {}".format(end - start))
        # plot.decisions(x, z_vector, y, p, alpha, b, include_observations=True, include_probs=True,logscale=False)
        # plot.argmax_sizes(prob_0, prob_1, p)
        # plot.a_i(a_vec_0,a_vec_1, b_vec_0, b_vec_1)
        plot.a_i_decisions(b_vec_0, b_vec_1, z_vector,p, y,  x, alpha, b, model=1)

        # x = np.linspace(0, 2, 500)
        # plt.plot(x, beta.pdf(x, alpha, b))
        # plt.grid(True)
        # plt.show()

    elif MasterFlag == '3: Calculate z_i for model 2':
        alpha = 15
        b = 6
        start=time.time()
        z_vector, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1 = model2.z_i(n=25, alpha=alpha, b=b)
        end=time.time()
        print("Time elapsed = {}".format(end-start))
        b_cont=np.zeros(25)
        # b_cont += b_vec_0
        # for i in range(10):
        #     z_vector, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1 = model2.z_i(n=25, alpha=alpha, b=b)
        #     plt.plot(b_vec_0)
        #     b_cont += b_vec_0
        # plt.show()
        # b_cont /= 10
        # plt.plot(b_cont)
        # plt.show()
        plot.a_i_decisions(b_vec_0, b_vec_1, z_vector, p, y, x, alpha, b, model=2)
        # plot.decisions(x, z_vector, y, p, alpha, b, include_observations=False, include_probs=True, grid=False)
        # plot.argmax_sizes(prob_0, prob_1, p)
        # plot.a_i(a_vec_0, a_vec_1, b_vec_0, b_vec_1)
        # print(b_vec_0)
        # print(b_vec_1)
        # a_i_dist=np.load("a_i_distr.npy")
        # plt.hist(a_i_dist, bins=40)
        # plt.show()

    elif MasterFlag == '4: Model 1: How fast does a cascade occur?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")
        a_i_matrix_0 = np.load("data/model1/n_last_cascade/a_matrix_0_fixed_expectation_p_i_0.55_various_variances.npy")
        a_i_matrix_1 = np.load("data/model1/n_last_cascade/a_matrix_1_fixed_expectation_p_i_0.55_various_variances.npy")
        median_cascade = np.load("data/model1/n_last_cascade/n_last_cascade_median.npy")
        avg_cascade = np.load("data/model1/n_last_cascade/n_last_cascade_avg.npy")
        distribution = np.load("data/model1/n_last_cascade/distribution_n_until_last_cascade.npy")
        plot.distribution(distribution, variances, mean_values[1], cumulative=True, histtype="step")
        plot.time_to_cascade(median_cascade, avg_cascade, mean_values, variances)
        plot.a_i_values_fixed_p_i(a_i_matrix_0,a_i_matrix_1,0.55, variances)

    elif MasterFlag == '5: Model 2: How fast does a cascade occur?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")
        median_cascade = np.load("data/model2/n_until_last_cascade/n_until_last_cascade_median.npy")
        avg_cascade = np.load("data/model2/n_until_last_cascade/n_until_last_cascade_avg.npy")
        plot.time_to_cascade(median_cascade, avg_cascade, mean_values, variances)
        a_i_matrix_0 = np.load("data/model2/n_until_last_cascade/a_matrix/a_matrix_0_fixed_expectation_different_variances.npy")
        a_i_matrix_1 = np.load("data/model2/n_until_last_cascade/a_matrix/a_matrix_1_fixed_expectation_different_variances.npy")
        plot.a_i_values_fixed_p_i(a_i_matrix_0, a_i_matrix_1, 0.55, variances)
        distribution = np.load("data/model2/n_until_last_cascade/distribution_n_until_last_cascade.npy")
        plot.distribution(distribution, variances, mean_values[1], cumulative=False, histtype="bar")

    elif MasterFlag == '6: How fast does a cascade occur?':
        print("Model 1 and model 2")
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")
        #Load data for Model 1
        a_i_matrix_0_model1 = np.load("data/model1/n_last_cascade/a_matrix_0_fixed_expectation_p_i_0.55_various_variances.npy")
        a_i_matrix_1_model1 = np.load("data/model1/n_last_cascade/a_matrix_1_fixed_expectation_p_i_0.55_various_variances.npy")
        median_cascade_model1 = np.load("data/model1/n_last_cascade/n_last_cascade_median.npy")
        avg_cascade_model1 = np.load("data/model1/n_last_cascade/n_last_cascade_avg.npy")
        distribution_model1 = np.load("data/model1/n_last_cascade/distribution_n_until_last_cascade.npy")
        # Load data for Model 2
        median_cascade_model2 = np.load("data/model2/n_until_last_cascade/n_until_last_cascade_median.npy")
        avg_cascade_model_2 = np.load("data/model2/n_until_last_cascade/n_until_last_cascade_avg.npy")
        distribution_model2 = np.load("data/model2/n_until_last_cascade/distribution_n_until_last_cascade.npy")
        a_i_matrix_0_model2 = np.load(
            "data/model2/n_until_last_cascade/a_matrix/a_matrix_0_fixed_expectation_different_variances.npy")
        a_i_matrix_1_model2 = np.load(
            "data/model2/n_until_last_cascade/a_matrix/a_matrix_1_fixed_expectation_different_variances.npy")

        # Plot data

        plot.time_to_cascade(median_cascade_model1, avg_cascade_model1, mean_values, variances, median_cascade_model2, avg_cascade_model_2, compare=True)
        plot.distribution(distribution_model1, variances, mean_values[1], distribution_model2, cumulative=False, histtype="bar", compare=True, matrix_plot=True)
        plot.a_i_values_fixed_p_i(a_i_matrix_0_model1, a_i_matrix_1_model1, mean_values[1], variances, a_i_matrix_0_model2, a_i_matrix_1_model2, compare=True)

    elif MasterFlag == '7: Model 1: Is the cascade correct?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")
        last_cascade_correct = np.load("data/model1/correct_cascade/last_cascade_correct.npy")
        first_cascade_correct = np.load("data/model1/correct_cascade/first_cascade_correct.npy")
        total_fraction_wrong = np.load("data/model1/correct_cascade/total_fraction_wrong.npy")
        a_matrix_0 = np.load("data/model1/correct_cascade/a_matrix_0_fixed_var.npy")
        a_matrix_1 = np.load("data/model1/correct_cascade/a_matrix_1_fixed_var.npy")
        plot.correct_cascade(last_cascade_correct, first_cascade_correct, total_fraction_wrong, mean_values, variances)
        plot.a_i_fixed_variance(a_matrix_0, a_matrix_1, mean_values, variances[1])

    elif MasterFlag == '8: Model 2: Is the cascade correct?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")
        last_cascade_correct = np.load("data/model2/correct_cascade/last_cascade_correct.npy")
        first_cascade_correct = np.load("data/model2/correct_cascade/first_cascade_correct.npy")
        total_fraction_wrong = np.load("data/model2/correct_cascade/total_fraction_wrong.npy")
        a_matrix_0 = np.load("data/model2/correct_cascade/a_matrix/a_matrix_0_fixed_variance_different_exp.npy")
        a_matrix_1 = np.load("data/model2/correct_cascade/a_matrix/a_matrix_1_fixed_variance_different_exp.npy")
        plot.correct_cascade(last_cascade_correct, first_cascade_correct, total_fraction_wrong, mean_values, variances)
        plot.a_i_fixed_variance(a_matrix_0, a_matrix_1, mean_values, variances[1])

    elif MasterFlag == '9: Is the cascade correct?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")

        ## Load data from model 1:
        last_cascade_correct_model1 = np.load("data/model1/correct_cascade/last_cascade_correct.npy")
        first_cascade_correct_model1 = np.load("data/model1/correct_cascade/first_cascade_correct.npy")
        total_fraction_wrong_model1 = np.load("data/model1/correct_cascade/total_fraction_wrong.npy")
        a_matrix_0_model1 = np.load("data/model1/correct_cascade/a_matrix_0_fixed_var.npy")
        a_matrix_1_model1 = np.load("data/model1/correct_cascade/a_matrix_1_fixed_var.npy")
        ## Load data from model 2:
        last_cascade_correct_model2 = np.load("data/model2/correct_cascade/last_cascade_correct.npy")
        first_cascade_correct_model2 = np.load("data/model2/correct_cascade/first_cascade_correct.npy")
        total_fraction_wrong_model2 = np.load("data/model2/correct_cascade/total_fraction_wrong.npy")
        a_matrix_0_model2 = np.load("data/model2/correct_cascade/a_matrix/a_matrix_0_fixed_variance_different_exp.npy")
        a_matrix_1_model2 = np.load("data/model2/correct_cascade/a_matrix/a_matrix_1_fixed_variance_different_exp.npy")

        ## Plot data
        plot.correct_cascade(last_cascade_correct_model1, first_cascade_correct_model1, total_fraction_wrong_model1, mean_values, variances, last_cascade_correct_model2, first_cascade_correct_model2, total_fraction_wrong_model2, compare=True)
        plot.a_i_fixed_variance(a_matrix_0_model1, a_matrix_1_model1, mean_values, variances[1], a_matrix_0_model2, a_matrix_1_model2, compare=True)

    elif MasterFlag == '10: Model 1: Is the cascade stable?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")

        average_number_of_changes = np.load("data/model1/stable_cascade/average/number_of_changes_avg.npy")
        one_change = np.load("data/model1/stable_cascade/average/one_change.npy")
        average_wrong_changes = np.load("data/model1/stable_cascade/average/wrong_changes_avg.npy")
        average_new_cascades = np.load("data/model1/stable_cascade/average/new_cascades_avg.npy")
        average_singles = np.load("data/model1/stable_cascade/average/singles_avg.npy")

        median_number_of_changes = np.load("data/model1/stable_cascade/median/number_of_changes_median.npy")
        median_wrong_changes = np.load("data/model1/stable_cascade/median/wrong_changes_median.npy")
        median_new_cascades = np.load("data/model1/stable_cascade/median/new_cascades_median.npy")
        median_singles = np.load("data/model1/stable_cascade/median/singles_median.npy")

        plot.number_of_changes(average_number_of_changes, median_number_of_changes, mean_values, variances)
        plot.fraction_of_one_change(one_change, mean_values, variances)
        plot.wrong_cascades(average_wrong_changes, median_wrong_changes, mean_values, variances)
        plot.new_cascades(average_new_cascades, median_new_cascades, mean_values, variances)
        plot.singles(average_singles, median_singles, mean_values, average_singles)

    elif MasterFlag == '11 Model 2: Is the cascade stable?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")

        average_number_of_changes = np.load("data/model2/stable_cascade/avg/number_of_changes_avg.npy")
        one_change = np.load("data/model2/stable_cascade/avg/one_change.npy")
        average_wrong_changes = np.load("data/model2/stable_cascade/avg/wrong_cascade_avg.npy")
        average_new_cascades = np.load("data/model2/stable_cascade/avg/new_cascade_avg.npy")
        average_singles = np.load("data/model2/stable_cascade/avg/singles_avg.npy")

        median_number_of_changes = np.load("data/model2/stable_cascade/median/number_of_changes_median.npy")
        median_wrong_changes = np.load("data/model2/stable_cascade/median/wrong_cascade_median.npy")
        median_new_cascades = np.load("data/model2/stable_cascade/median/new_cascade_median.npy")
        median_singles = np.load("data/model2/stable_cascade/median/singles_median.npy")

        plot.number_of_changes(average_number_of_changes, median_number_of_changes, mean_values, variances)
        plot.fraction_of_one_change(one_change, mean_values, variances)
        plot.wrong_cascades(average_wrong_changes, median_wrong_changes, mean_values, variances)
        plot.new_cascades(average_new_cascades, median_new_cascades, mean_values, variances)
        plot.singles(average_singles, median_singles, mean_values, average_singles)

    elif MasterFlag == '12: Is the cascade stable?':
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")

        ## Load data from model 1:
        median_number_of_changes_model1 = np.load("data/model1/stable_cascade/median/number_of_changes_median.npy")
        median_wrong_changes_model1 = np.load("data/model1/stable_cascade/median/wrong_changes_median.npy")
        median_new_cascades_model1 = np.load("data/model1/stable_cascade/median/new_cascades_median.npy")
        median_singles_model1 = np.load("data/model1/stable_cascade/median/singles_median.npy")
        average_number_of_changes_model1 = np.load("data/model1/stable_cascade/average/number_of_changes_avg.npy")
        one_change_model1 = np.load("data/model1/stable_cascade/average/one_change.npy")
        average_wrong_changes_model1 = np.load("data/model1/stable_cascade/average/wrong_changes_avg.npy")
        average_new_cascades_model1 = np.load("data/model1/stable_cascade/average/new_cascades_avg.npy")
        average_singles_model1 = np.load("data/model1/stable_cascade/average/singles_avg.npy")

        ## Load data from model 2:
        average_number_of_changes_model2 = np.load("data_400/model2/stable_cascade/avg/number_of_changes_avg.npy")
        one_change_model2 = np.load("data_400/model2/stable_cascade/avg/one_change.npy")
        average_wrong_changes_model2 = np.load("data_400/model2/stable_cascade/avg/wrong_cascade_avg.npy")
        average_new_cascades_model2 = np.load("data_400/model2/stable_cascade/avg/new_cascade_avg.npy")
        average_singles_model2 = np.load("data_400/model2/stable_cascade/avg/singles_avg.npy")
        median_number_of_changes_model2 = np.load("data_400/model2/stable_cascade/median/number_of_changes_median.npy")
        median_wrong_changes_model2 = np.load("data_400/model2/stable_cascade/median/wrong_cascade_median.npy")
        median_new_cascades_model2 = np.load("data_400/model2/stable_cascade/median/new_cascade_median.npy")
        median_singles_model2 = np.load("data_400/model2/stable_cascade/median/singles_median.npy")

        ## Plot data
        plot.number_of_changes(average_number_of_changes_model1, median_number_of_changes_model1, mean_values, variances, average_number_of_changes_model2, median_number_of_changes_model2, compare=True)
        plot.fraction_of_one_change(one_change_model1, mean_values, variances, one_change_model2, compare=True)
        plot.wrong_cascades(average_wrong_changes_model1, median_wrong_changes_model1, mean_values, variances, average_wrong_changes_model2, median_wrong_changes_model2, compare=True)
        plot.new_cascades(average_new_cascades_model1, median_new_cascades_model1, mean_values, variances, average_new_cascades_model2, median_new_cascades_model2, compare=True)
        plot.singles(average_singles_model1, median_singles_model1, mean_values, variances, median_singles_model2, average_singles_model2, compare=True)

    elif MasterFlag == '13: Plots':
        alpha_values = np.load("variables/alpha_values.npy")
        beta_values = np.load("variables/beta_values.npy")
        mean_values = np.load("variables/mean_values.npy")
        variances = np.load("variables/variance_values.npy")
        last_cascade_correct_model1 = np.load("data/model1/correct_cascade/last_cascade_correct.npy")
        first_cascade_correct_model1 = np.load("data/model1/correct_cascade/first_cascade_correct.npy")
        last_cascade_correct_model2 = np.load("data/model2/correct_cascade/last_cascade_correct.npy")
        first_cascade_correct_model2 = np.load("data/model2/correct_cascade/first_cascade_correct.npy")
        changes_dist = np.load("number_of_changes_dist.npy")
        plt.hist(changes_dist, bins=20, density=True)
        plt.show()
        print(np.median(changes_dist))
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        for i in range(3):
            ax[0].plot((np.abs(last_cascade_correct_model1[i,:] - first_cascade_correct_model1[i, :])))
            ax[1].plot((np.abs(last_cascade_correct_model2[i, :] - first_cascade_correct_model2[i, :])))
        plt.show()

        plot.numerical_setup(variances, mean_values, alpha_values, beta_values)

    elif MasterFlag == '14: Test':
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