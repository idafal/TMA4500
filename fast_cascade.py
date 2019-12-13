import numpy as np
import time
import glob

import simulation as model1
import simulation_model2 as model2
import utility_functions as util
"""
Script to check how fast the final cascade occurs.
- Remember to let the true value x=0
- 
"""
## Todo:
# - add variances and expectations to written alpha and beta values
# - add script for model 2

model=1
write_to_file = True
run_merge_files = True

print("Loading values for parameters alpha and beta")
alpha_values = np.load("variables/alpha_values.npy")
beta_values = np.load("variables/beta_values.npy")

n_variances = len(beta_values[0, :])
n_expected_values = len(beta_values[:, 0])
print("Model: {}".format(model))
# Model 1
if model == 1:
    n = 1000  # Individuals in the chain of decisions
    n_runs = 200  # numbers to average over

    print("Number of expected values to calculate: {}".format(n_expected_values))
    print("Number of variances: {}".format(n_variances))
    print("Averaging over {} simulations".format(n_runs))
    print("Running chains of n = {} individuals".format(n))

    ## Fast cascade
    a_matrix_0_fixed_expectation = np.zeros((n_variances, n))  # Container for a_i values given x=0
    a_matrix_1_fixed_expectation = np.zeros((n_variances, n))  # Container for a_i values given x=1
    z_vector = np.zeros((n_runs, n))  # Container for z-values
    n_until_last_cascade_avg = np.zeros(
        (n_variances, n_expected_values))  # Container for number of individuals until cascade occurs
    n_until_last_cascade_median = np.zeros(
        (n_variances, n_expected_values))  # Container for median number of individuals until cascade occurs

    fixed_expectation_last_cascade = np.zeros((n_variances, n_runs))
    a_i_dist = np.zeros(n_runs)


    ## Correct cascade
    a_matrix_0_fixed_variance = np.zeros((n_expected_values, n))  # Container for a_i values given x=0
    a_matrix_1_fixed_variance = np.zeros((n_expected_values, n))  # Container for a_i values given x=1
    last_cascade_correct = np.zeros((n_variances, n_expected_values))
    first_cascade_correct = np.zeros((n_variances, n_expected_values))
    total_fraction_wrong = np.zeros((n_variances, n_expected_values))

    ## Stable cascade
    number_of_changes_avg = np.zeros((n_variances, n_expected_values))
    wrong_changes_avg = np.zeros((n_variances, n_expected_values))
    one_change = np.zeros((n_variances, n_expected_values))
    new_cascades_avg = np.zeros((n_variances, n_expected_values))
    singles_avg = np.zeros((n_variances, n_expected_values))

    number_of_changes_median = np.zeros((n_variances, n_expected_values))
    wrong_changes_median = np.zeros((n_variances, n_expected_values))
    new_cascades_median = np.zeros((n_variances, n_expected_values))
    singles_median = np.zeros((n_variances, n_expected_values))



    start = time.time()

    print("Calculating number of individuals until last cascade occurs for model 1")
    for j in range(n_variances):
        print("Calculating for variance number {}".format(j+1))
        for k in range(n_expected_values):
            print("Calculating for expected value number: {}".format(k))
            individuals_until_cascade = np.zeros(n_runs)

            n_change_current = np.zeros(n_runs)
            wrong_change_current = np.zeros(n_runs)
            new_cascades_current = np.zeros(n_runs)
            singles_current = np.zeros(n_runs)
            for i in range(n_runs):
                z_vector, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1 = model1.z(n=n, alpha=alpha_values[k,j], beta=beta_values[k,j])

                ## Find fast cascade
                last_value = z_vector[-1]
                last_wrong_index = np.argwhere(z_vector != last_value)
                if last_wrong_index.size == 0:
                    n_until_last_cascade_avg[j,k] += 0
                    individuals_until_cascade[i] = 0

                else:
                    n_until_last_cascade_avg[j,k] += last_wrong_index[-1] + 1
                    individuals_until_cascade[i] = last_wrong_index[-1] + 1
                    if k == 1:
                        fixed_expectation_last_cascade[j, i] = last_wrong_index[-1] + 1
                if k == 1:
                    a_matrix_0_fixed_expectation[j, :] += b_vec_0
                    a_matrix_1_fixed_expectation[j, :] += b_vec_1


                ## Correct cascade
                last_cascade_correct[j, k] += 1 - last_value
                first_cascade = util.get_n_in_row(z_vector, n=10)
                if first_cascade != -1:
                    first_cascade_correct[j, k] += (1 - first_cascade)
                total_fraction_wrong[j, k] += np.sum(z_vector)
                if j == 1:
                    a_matrix_0_fixed_variance[k, :] += b_vec_0
                    a_matrix_1_fixed_variance[k, :] += b_vec_1

                ## Stable cascade
                groups = util.get_groups(z_vector)

                if len(groups) !=  1:

                    n_change_current[i], one_change_c, wrong_change_current[i], new_cascades_current[i],  singles_current[i] = util.change_in_cascade(groups, n=10)
                    number_of_changes_avg[j, k] += n_change_current[i]
                    one_change[j, k] += one_change_c
                    wrong_changes_avg[j, k] += wrong_change_current[i]
                    new_cascades_avg[j, k] += new_cascades_current[i]
                    singles_avg[j, k] += singles_current[i]


            if k==1 and j==1:
                np.save("a_i_distr.npy", a_i_dist, allow_pickle=True)
                np.save("number_of_changes_dist.npy", n_change_current, allow_pickle=True)
                print("a_i_distr.npy successfully written to file")

            n_until_last_cascade_median[j,k] = np.median(individuals_until_cascade)

            number_of_changes_median[j, k] = np.median(n_change_current)
            wrong_changes_median[j,k] = np.median(wrong_change_current)
            new_cascades_avg[j, k] = np.median(new_cascades_current)
            singles_median[j, k] = np.median(singles_current)

    n_until_last_cascade_avg = n_until_last_cascade_avg / n_runs
    a_matrix_0_fixed_expectation = a_matrix_0_fixed_expectation / n_runs
    a_matrix_1_fixed_expectation = a_matrix_1_fixed_expectation / n_runs

    last_cascade_correct = last_cascade_correct / n_runs
    first_cascade_correct = first_cascade_correct / n_runs
    total_fraction_wrong = total_fraction_wrong / n_runs
    a_matrix_0_fixed_variance = a_matrix_0_fixed_variance / n_runs
    a_matrix_1_fixed_variance = a_matrix_1_fixed_variance / n_runs

    number_of_changes_avg = number_of_changes_avg / n_runs
    one_change = one_change / n_runs
    wrong_changes_avg = wrong_changes_avg / n_runs
    new_cascades_avg = new_cascades_avg / n_runs
    singles_avg = singles_avg / n_runs
    if write_to_file:
        # Writing fast cascade
        np.save("data/model1/n_last_cascade/a_matrix_0_fixed_expectation_p_i_0.55_various_variances.npy", a_matrix_0_fixed_expectation, allow_pickle=True)
        np.save("data/model1/n_last_cascade/a_matrix_1_fixed_expectation_p_i_0.55_various_variances.npy", a_matrix_1_fixed_expectation, allow_pickle=True)
        np.save("data/model1/n_last_cascade/n_last_cascade_median.npy", n_until_last_cascade_median, allow_pickle=True)
        np.save("data/model1/n_last_cascade/n_last_cascade_avg.npy", n_until_last_cascade_avg, allow_pickle=True)
        np.save("data/model1/n_last_cascade/distribution_n_until_last_cascade",fixed_expectation_last_cascade, allow_pickle=True)
        # Writing correct cascade
        np.save("data/model1/correct_cascade/last_cascade_correct.npy", last_cascade_correct, allow_pickle=True)
        np.save("data/model1/correct_cascade/first_cascade_correct.npy", first_cascade_correct, allow_pickle=True)
        np.save("data/model1/correct_cascade/total_fraction_wrong.npy", total_fraction_wrong, allow_pickle=True)
        np.save("data/model1/correct_cascade/a_matrix_0_fixed_var.npy", a_matrix_0_fixed_variance, allow_pickle=True)
        np.save("data/model1/correct_cascade/a_matrix_1_fixed_var.npy", a_matrix_1_fixed_variance, allow_pickle=True)
        # Writing stable cascade
        np.save("data/model1/stable_cascade/average/number_of_changes_avg.npy", number_of_changes_avg,allow_pickle=True)
        np.save("data/model1/stable_cascade/average/one_change.npy", one_change, allow_pickle=True)
        np.save("data/model1/stable_cascade/average/wrong_changes_avg.npy", wrong_changes_avg, allow_pickle=True)
        np.save("data/model1/stable_cascade/average/new_cascades_avg.npy", new_cascades_avg)
        np.save("data/model1/stable_cascade/average/singles_avg.npy", singles_avg, allow_pickle=True)
        np.save("data/model1/stable_cascade/median/number_of_changes_median.npy", number_of_changes_median, allow_pickle=True)
        np.save("data/model1/stable_cascade/median/wrong_changes_median.npy", wrong_changes_median, allow_pickle=True)
        np.save("data/model1/stable_cascade/median/new_cascades_median.npy", new_cascades_median, allow_pickle=True)
        np.save("data/model1/stable_cascade/median/singles_median.npy", singles_median, allow_pickle=True)

        print("Data successfully written")
    end = time.time()
    print("Time elsapsed = {}".format(end - start))

if model == 2:
    n = 1000
    n_runs = 200
    variance = 1
    expectation = 1

    print("Calculating number of individuals until last cascade occurs for model 2")
    print("Averaging over {} simulations".format(n_runs))
    print("Running chains of n = {} individuals".format(n))

    ## Fast cascade
    n_until_last_cascade_avg = 0 # Keeping track of the average number of individuals until cascade
    individuals_until_cascade = np.zeros(n_runs) # Keeping track of the median number of individuals until cacsade
    a_matrix_0_fixed_expectation = np.zeros(n)
    a_matrix_1_fixed_expectation = np.zeros(n)
    distribution_n_last_cascade = np.zeros(n_runs)

    ## Correct cascade
    a_matrix_0_fixed_variance = np.zeros(n)  # Container for a_i values given x=0
    a_matrix_1_fixed_variance = np.zeros(n)  # Container for a_i values given x=1
    last_cascade_correct = 0
    first_cascade_correct = 0
    total_fraction_wrong = 0

    ## Stable cascade
    number_of_changes_avg = 0
    number_of_changes_median = np.zeros(n_runs)
    wrong_changes_avg = 0
    wrong_changes_median = np.zeros(n_runs)
    new_cascades_avg = 0
    new_cascades_median = np.zeros(n_runs)
    singles_avg = 0
    singles_median = np.zeros(n_runs)
    one_change = 0

    start = time.time()
    for i in range(n_runs):
        print("Simulation {} out of {}".format(i+1, n_runs))
        z_vector, x, y, p, prob_0, prob_1, a_vec_0, a_vec_1, b_vec_0, b_vec_1 = model2.z_i(n=n, alpha=alpha_values[expectation, variance],
                                                                                         b=beta_values[expectation, variance])
        ## Fast cascade
        last_value = z_vector[-1]
        last_wrong_index = np.argwhere(z_vector != last_value)
        if (last_wrong_index.size == 0):
            n_until_last_cascade_avg += 0
            individuals_until_cascade[i] = 0
        else:
            n_until_last_cascade_avg += last_wrong_index[-1] + 1 # +1 as python is zero-indexed
            individuals_until_cascade[i] = last_wrong_index[-1] + 1
            distribution_n_last_cascade[i] = last_wrong_index[-1] + 1

        if expectation == 1:
            a_matrix_0_fixed_expectation += b_vec_0
            a_matrix_1_fixed_expectation += b_vec_1


        ## Correct cascade
        last_cascade_correct += 1 - last_value # assuming true value is 0
        first_cascade = util.get_n_in_row(z_vector, n=10)
        if first_cascade != -1:
            first_cascade_correct += (1 - first_cascade)
        total_fraction_wrong += np.sum(z_vector)

        if variance == 1:
            a_matrix_0_fixed_variance += b_vec_0
            a_matrix_1_fixed_variance += b_vec_1

        ## Stable cascade
        groups = util.get_groups(z_vector)
        if len(groups) != 1:
            number_of_changes_median[i], one_change_c, wrong_changes_median[i], new_cascades_median[i], singles_median[i] = util.change_in_cascade(groups, n=10)
            number_of_changes_avg += number_of_changes_median[i]
            one_change += one_change_c
            wrong_changes_avg += wrong_changes_median[i]
            print(wrong_changes_median[i])
            new_cascades_avg += new_cascades_median[i]
            singles_avg += singles_median[i]

    n_until_last_cascade_avg = n_until_last_cascade_avg / n_runs
    median_last_cascade = np.median(individuals_until_cascade)
    last_cascade_correct = last_cascade_correct / n_runs
    first_cascade_correct = first_cascade_correct / n_runs
    total_fraction_wrong = total_fraction_wrong / n_runs
    if expectation == 1 and variance == 1:
        np.save("number_of_changes_dist_model2.npy", number_of_changes_median, allow_pickle=True)
    number_of_changes_median = np.median(number_of_changes_median)
    number_of_changes_avg = number_of_changes_avg / n_runs
    one_change = one_change / n_runs
    wrong_changes_median = np.median(wrong_changes_median)
    wrong_changes_avg = wrong_changes_avg / n_runs
    new_cascades_median = np.median(new_cascades_median)
    new_cascades_avg = new_cascades_avg / n_runs
    singles_median = np.median(singles_median)
    singles_avg = singles_avg / n_runs


    end = time.time()
    print("Time elsapsed = {}".format(end - start))

    if write_to_file == True:
        print("Writing data to file")

        ## Last cascade
        filename_average = "data/model2/n_until_last_cascade/average/average_var_" + str(variance) + "_exp_" + str(expectation)
        filename_median = "data/model2/n_until_last_cascade/median/median_var_" + str(variance) + "_exp_" + str(expectation)

        ## Correct cascade
        filename_last_correct = "data/model2/correct_cascade/last_cacsade/last_cascade_var_" + str(variance) + "_exp_" + str(expectation)
        filename_first_correct = "data/model2/correct_cascade/first_cascade/first_cascade_var_" + str(
            variance) + "_exp_" + str(expectation)
        filename_total_fraction_wrong = "data/model2/correct_cascade/total_wrong/total_wrong_var" + str(variance) + "_exp_" +str(expectation)

        ## Stable cascade
        avg_number_of_changes_filename = "data/model2/stable_cascade/avg/number_of_changes/number_of_changes_"+ str(variance) + "_exp_" + str(expectation)
        avg_one_change_filename = "data/model2/stable_cascade/avg/one_change/one_change_"+ str(variance) + "_exp_" + str(expectation)
        avg_wrong_change_filename = "data/model2/stable_cascade/avg/wrong_change/wrong_change_"+ str(variance) + "_exp_" + str(expectation)
        avg_new_cascade_filename = "data/model2/stable_cascade/avg/new_cascade/new_cascade_"+ str(variance) + "_exp_" + str(expectation)
        avg_singles_filename = "data/model2/stable_cascade/avg/singles/singles_"+ str(variance) + "_exp_" + str(expectation)
        median_number_of_changes_filename = "data/model2/stable_cascade/median/number_of_changes/number_of_changes_"+ str(variance) + "_exp_" + str(expectation)
        median_wrong_change_filename = "data/model2/stable_cascade/median/wrong_change/wrong_change_"+ str(variance) + "_exp_" + str(expectation)
        median_new_cascade_filename = "data/model2/stable_cascade/median/new_cascade/new_cascade_"+ str(variance) + "_exp_" + str(expectation)
        median_singles_filename = "data/model2/stable_cascade/median/singles/singles_"+ str(variance) + "_exp_" + str(expectation)

        np.save(filename_average,n_until_last_cascade_avg, allow_pickle=True)
        np.save(filename_median, median_last_cascade, allow_pickle=True)
        np.save(filename_last_correct, last_cascade_correct, allow_pickle=True)
        np.save(filename_first_correct, first_cascade_correct, allow_pickle=True)
        np.save(filename_total_fraction_wrong,  total_fraction_wrong, allow_pickle=True)

        np.save(avg_number_of_changes_filename, number_of_changes_avg, allow_pickle=True)
        np.save(median_number_of_changes_filename, number_of_changes_median, allow_pickle=True)
        np.save(avg_one_change_filename, one_change, allow_pickle=True)
        np.save(median_wrong_change_filename, wrong_changes_median, allow_pickle=True)
        np.save(avg_wrong_change_filename, wrong_changes_avg, allow_pickle=True)
        np.save(avg_new_cascade_filename, new_cascades_avg, allow_pickle=True)
        np.save(median_new_cascade_filename, new_cascades_median, allow_pickle=True)
        np.save(avg_singles_filename, singles_avg, allow_pickle=True)
        np.save(median_singles_filename, singles_median, allow_pickle=True)

        if expectation == 1:
            a_matrix_0_fixed_expectation = a_matrix_0_fixed_expectation / n_runs
            a_matrix_1_fixed_expectation = a_matrix_1_fixed_expectation / n_runs
            filename_a_0 = "data/model2/n_until_last_cascade/a_matrix/a0/a_0_var_" + str(variance)
            filename_a_1 = "data/model2/n_until_last_cascade/a_matrix/a1/a_1_var_" + str(variance)
            np.save(filename_a_0, a_matrix_0_fixed_expectation, allow_pickle=True)
            np.save(filename_a_1, a_matrix_1_fixed_expectation, allow_pickle=True)
            filename_distr = "data/model2/n_until_last_cascade/distribution/distribution_n_until_last_cascade_exp_" + str(expectation) + "_var_" + str(variance)
            np.save(filename_distr, distribution_n_last_cascade, allow_pickle=True)
        if variance == 1:
            a_matrix_0_fixed_variance = a_matrix_0_fixed_variance / n_runs
            a_matrix_1_fixed_variance = a_matrix_1_fixed_variance / n_runs
            filename_a_0_fixed_var = "data/model2/correct_cascade/a_matrix/a_0/a_0_exp_" + str(expectation)
            filename_a_1_fixed_var = "data/model2/correct_cascade/a_matrix/a_1/a_1_exp_" + str(expectation)
            np.save(filename_a_0_fixed_var, a_matrix_0_fixed_variance, allow_pickle=True)
            np.save(filename_a_1_fixed_var, a_matrix_1_fixed_variance, allow_pickle=True)
        print("Data successfully written to file.")


def model2_merge_files(n_variances, n_expectations, n, n_runs):
    """
    Function to merge all subfiles from model 2. Write the result to file.
    :param n_variances: Number of total variances used
    :param n_expectations: Number of total expectations used
    :param n: Numebr of individuals in one chain
    """
    ## Median and average number of decisions until a cascade occurs

    path_avg = "data/model2/n_until_last_cascade/average/"
    path_median = "data/model2/n_until_last_cascade/median/"
    path_distr = "data/model2/n_until_last_cascade/distribution/"
    files_avg = glob.glob(path_avg + "*.npy")
    files_median = glob.glob(path_median + "*.npy")
    files_distr = glob.glob(path_distr + "*.npy" )
    n_until_last_cascade_avg = np.zeros(
        (n_variances, n_expectations))  # Container for number of individuals until cascade occurs
    n_until_last_cascade_median = np.zeros(
        (n_variances, n_expectations))  # Container for median number of individuals until cascade occurs

    fixed_expectation_last_cascade = np.zeros((n_variances, n_runs))

    for i in range(len(files_distr)):
        cur_var = int(files_distr[i][-5])
        fixed_expectation_last_cascade[cur_var, :] = np.load(str(files_distr[i]))
    np.save("data/model2/n_until_last_cascade/distribution_n_until_last_cascade", fixed_expectation_last_cascade, allow_pickle=True)

    for i in range(len(files_avg)):
        current_variance_avg = int(files_avg[i][-11])
        current_expectation_avg = int(files_avg[i][-5])
        current_variance_median = int(files_median[i][-11])
        current_expectation_median = int(files_median[i][-5])
        n_until_last_cascade_avg[current_variance_avg, current_expectation_avg] = np.load(str(files_avg[i]))
        n_until_last_cascade_median[current_variance_median, current_expectation_median] = np.load(files_median[i])

    np.save("data/model2/n_until_last_cascade/n_until_last_cascade_avg.npy",n_until_last_cascade_avg, allow_pickle=True)
    np.save("data/model2/n_until_last_cascade/n_until_last_cascade_median.npy", n_until_last_cascade_median, allow_pickle=True)

    ## Averaged a_i values for a fixed value of the mean and for different variances
    path_a_matrix_0_fixed_expectation = "data/model2/n_until_last_cascade/a_matrix/a0/"
    path_a_matrix_1_fixed_expectation = "data/model2/n_until_last_cascade/a_matrix/a1/"
    path_a_matrix_0_fixed_variance = "data/model2/correct_cascade/a_matrix/a_0/"
    path_a_matrix_1_fixed_variance = "data/model2/correct_cascade/a_matrix/a_1/"

    files_a_matrix_0_fixed_expectation = glob.glob(path_a_matrix_0_fixed_expectation + "*.npy")
    files_a_matrix_1_fixed_expectation = glob.glob(path_a_matrix_1_fixed_expectation + "*.npy")

    a_matrix_0_fixed_expectation = np.zeros((n_variances, n))  # Container for a_i values given x=0
    a_matrix_1_fixed_expectation = np.zeros((n_variances, n))  # Container for a_i values given x=1
    for i in range(len(files_a_matrix_0_fixed_expectation)):
        var_a0 = int(files_a_matrix_0_fixed_expectation[i][-5])
        var_a1 = int(files_a_matrix_1_fixed_expectation[i][-5])
        a_matrix_0_fixed_expectation[var_a0, :] = np.load(str(files_a_matrix_0_fixed_expectation[i]))
        a_matrix_1_fixed_expectation[var_a1, :] = np.load(str(files_a_matrix_1_fixed_expectation[i]))
    np.save("data/model2/n_until_last_cascade/a_matrix/a_matrix_0_fixed_expectation_different_variances.npy", a_matrix_0_fixed_expectation, allow_pickle=True)
    np.save("data/model2/n_until_last_cascade/a_matrix/a_matrix_1_fixed_expectation_different_variances.npy", a_matrix_1_fixed_expectation, allow_pickle=True)

    path_last = "data/model2/correct_cascade/last_cacsade/"
    path_first = "data/model2/correct_cascade/first_cascade/"
    path_total_wrong = "data/model2/correct_cascade/total_wrong/"
    files_last = glob.glob(path_last + "*.npy")
    files_first = glob.glob(path_first + "*.npy")
    files_total_wrong = glob.glob(path_total_wrong + "*.npy")

    last_cascade_correct = np.zeros((n_variances, n_expected_values))
    first_cascade_correct = np.zeros((n_variances, n_expected_values))
    total_fraction_wrong = np.zeros((n_variances, n_expected_values))

    for i in range(len(files_last)):
        current_variance_last = int(files_last[i][-11])
        current_expectation_last = int(files_last[i][-5])
        current_variance_first = int(files_first[i][-11])
        current_expectation_first = int(files_first[i][-5])
        current_variance_total_wrong = int(files_total_wrong[i][-11])
        current_expectation_total_wrong = int(files_total_wrong[i][-5])

        last_cascade_correct[current_variance_last, current_expectation_last] = np.load(str(files_last[i]))
        first_cascade_correct[current_variance_first, current_expectation_first] = np.load(files_first[i])
        total_fraction_wrong[current_variance_total_wrong, current_expectation_total_wrong] = np.load(files_total_wrong[i])

    np.save("data/model2/correct_cascade/last_cascade_correct.npy", last_cascade_correct, allow_pickle=True)
    np.save("data/model2/correct_cascade/first_cascade_correct.npy", first_cascade_correct, allow_pickle=True)
    np.save("data/model2/correct_cascade/total_fraction_wrong.npy", total_fraction_wrong, allow_pickle=True)

    ## a_matrix_fixed_variance
    files_a_matrix_0_fixed_variance = glob.glob(path_a_matrix_0_fixed_variance + "*.npy")
    files_a_matrix_1_fixed_variance = glob.glob(path_a_matrix_1_fixed_variance + "*.npy")

    a_matrix_0_fixed_variance = np.zeros((n_expectations, n))  # Container for a_i values given x=0
    a_matrix_1_fixed_variance = np.zeros((n_expectations, n))  # Container for a_i values given x=1
    for i in range(len(files_a_matrix_0_fixed_variance)):
        exp_a0 = int(files_a_matrix_0_fixed_variance[i][-5])
        exp_a1 = int(files_a_matrix_1_fixed_variance[i][-5])
        a_matrix_0_fixed_variance[exp_a0, :] = np.load(str(files_a_matrix_0_fixed_variance[i]))
        a_matrix_1_fixed_variance[exp_a1, :] = np.load(str(files_a_matrix_1_fixed_variance[i]))
    np.save("data/model2/correct_cascade/a_matrix/a_matrix_0_fixed_variance_different_exp.npy",
            a_matrix_0_fixed_variance, allow_pickle=True)
    np.save("data/model2/correct_cascade/a_matrix/a_matrix_1_fixed_variance_different_exp.npy",
            a_matrix_1_fixed_variance, allow_pickle=True)

    path_avg = "data/model2/stable_cascade/avg/"
    path_median = "data/model2/stable_cascade/median/"
    folder_list = ["new_cascade/", "number_of_changes/", "singles/", "wrong_cascade/"]
    for folder in folder_list:
        avg_container = np.zeros((n_variances, n_expectations))
        median_container = np.zeros((n_variances, n_expectations))
        full_path_avg = str(path_avg) + str(folder)
        avg_files = glob.glob(full_path_avg + "*.npy")
        full_path_median = str(path_median) + str(folder)
        median_files = glob.glob(full_path_median + "*.npy")
        for i in range(len(avg_files)):
            current_variance = int(avg_files[i][-11])
            current_expectation = int(avg_files[i][-5])
            avg_container[current_variance, current_expectation] = np.load(str(avg_files[i]))
        for i in range(len(median_files)):
            current_variance = int(median_files[i][-11])
            current_expectation = int(median_files[i][-5])
            median_container[current_variance, current_expectation] = np.load(str(median_files[i]))
        final_avg = path_avg + str(folder[:-1]) + "_avg.npy"
        final_median = path_median + str(folder[:-1]) + "_median.npy"
        np.save(final_avg, avg_container, allow_pickle=True)
        np.save(final_median, median_container, allow_pickle=True)

    one_change_path = "data/model2/stable_cascade/avg/one_change/"
    files = glob.glob(one_change_path + "*.npy")
    one_change = np.zeros((n_variances, n_expected_values))
    for i in range(len(files)):
        current_var = int(files[i][-11])
        current_exp = int(files[i][-5])
        one_change[current_var, current_exp] = np.load(str(files[i]))
    np.save("data/model2/stable_cascade/avg/one_change.npy", one_change, allow_pickle=True)

if run_merge_files:
    model2_merge_files(3,10, 1000, n_runs=400)
