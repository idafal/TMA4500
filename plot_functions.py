import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import beta

def a_i_decisions(a_i_0, a_i_1, decisions, probabilities,observations, x ,alpha,b, model, compare=False):
    if compare == False:
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle(r'Simulated chain of decisions for Model {} with true value $X= {}$ and parameters $\alpha = {}$, $\beta = {}$'.format(model, x, alpha, b), fontsize=22)
        axs[0].plot(decisions)
        axs[0].plot(decisions, 'bo--', label=r"Decisions $\bf{z}$")
        axs[0].plot(observations, color='grey', marker='o', linestyle='dashed', label=r"Private signals $\bf{y}$", alpha=0.65)
        axs[0].set_yticks([0,1])
        # axs[0].set_ylabel("Decision", fontsize=12)
        # axs[0].plot(probabilities, label=r"$P(Y=x|X=x)$", alpha=0.6)
        axs[2].plot(a_i_0, label = r"$a_{i,0}$")
        axs[2].plot(a_i_1, label = r"$a_{i,1}$")
        axs[0].legend(fontsize=15, loc=2)
        axs[1].plot(probabilities, label=r"Private signal $p_i$", alpha=0.6)
        axs[0].set_ylabel(r"Outcome", fontsize=20)
        axs[1].set_ylabel(r"$P(Y=x|X=x)$", fontsize=20)
        axs[1].legend(fontsize=15)
        axs[2].set_xlabel("Individuals", fontsize=20)
        axs[2].set_ylabel(r"$P(Z_i=z_i|x, \bf{z}_{\bf{i-\bf{1}}})$", fontsize=20)
        axs[2].legend(fontsize=15)
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        plt.show()
    if compare == True:
        print("Test")
    return

def decisions(x, z_vector, observations,p, alpha, b, include_observations=False, include_probs=False, block=True, grid=False, logscale=False):

    if logscale == True:
        plt.semilogx(z_vector)
        plt.semilogx(z_vector, 'bo--', label="Decision (z)")
    else:
        plt.plot(z_vector)
        plt.plot(z_vector, 'bo--', label="Decision (z)")
    if include_observations:
        if logscale==True:
            plt.semilogx(observations, color='grey', marker='o', linestyle='dashed', label="Observation (y)", alpha=0.5)
        plt.plot(observations, color='grey', marker='o', linestyle='dashed', label="Observation (y)", alpha=0.5)

    plt.xlabel("Individual")
    if include_probs:
        if logscale==True:
            plt.semilogx(p, label=r"Probability of observing correct $P(Y=x|X=x)$", alpha=0.7)
        plt.plot(p, label=r"Probability of observing correct $P(Y=x|X=x)$", alpha=0.7)
    plt.legend()
    if not include_probs:
        plt.yticks([0,1])
    plt.grid(grid)
    plt.title(r"True value: x = {}, parameters in beta-dist: $\alpha = {}$, $\beta = {}$".format(x, alpha, b))
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

def a_i(a_vec_0, a_vec_1, b_vec_0, b_vec_1):
    plt.plot(b_vec_0, label = r"$P(Z_i=z_i|X=0, \mathbf{z}_i)$")
    plt.plot(b_vec_1, label=r"$P(Z_i=z_i|X=1, \mathbf{z}_i)$")
    # plt.plot(a_vec_0, label=r"$a_i(\mathbf{z}_i, \mathbf{p}_i, x=0)$")
    # plt.plot(a_vec_1, label=r"$a_i(\mathbf{z}_i, \mathbf{p}_i, x=1)$")
    plt.legend()
    plt.show()

def time_to_cascade(median_cascade, avg_cascade, p_values, variances, model2_median=0, model2_average=0, compare=False):

    if compare==False:
        for i in range(len(variances)):
            plt.plot(p_values, median_cascade[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Number of individuals")
        plt.title(r"Median time until a cascade occurs")
        plt.legend()
        plt.show()
        for i in range(len(variances)):
            plt.plot(p_values, avg_cascade[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Number of individuals")
        plt.title(r"Average time until a cascade occurs")
        plt.legend()
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=1,ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].plot(p_values,median_cascade[i,:], marker=".", linewidth=3,markersize=13, alpha=0.8, label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(p_values,model2_median[i, :], marker=".",linewidth=3, markersize=13,alpha=0.8, label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[0].set_xlabel(r"E[$p_i$]", fontsize=20)
            ax[1].set_xlabel(r"E[$p_i$]", fontsize=20)
            ax[0].set_ylabel(r"Number of individuals", fontsize=15)
            # ax[1].set_ylabel(r"Number of individuals")
            ax[0].legend(fontsize=15, loc="upper center")
            ax[1].legend(fontsize=15)
            ax[0].set_title("Model 1", fontsize=20)
            ax[1].set_title("Model 2", fontsize=20)
            ax[0].grid(True)
            ax[1].grid(True)
        plt.suptitle(r"Median time until a cascade occurs", fontsize=25)
        plt.show()
        fig, ax = plt.subplots(nrows=1,ncols=2)
        for i in range(len(variances)):
            ax[0].plot(p_values, avg_cascade[i,:],label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(p_values, model2_average[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[0].set_xlabel(r"E[$p_i$]")
            ax[1].set_xlabel(r"E[$p_i$]")
            ax[0].set_ylabel(r"Number of individuals")
            ax[1].set_ylabel(r"Number of individuals")
            ax[0].legend(fontsize=15)
            ax[1].legend(fontsize=15)
            ax[0].set_title("Model 1")
            ax[1].set_title("Model 2")
        plt.suptitle(r"Average time until a cascade occurs")
        plt.show()

def a_i_values_fixed_p_i(a_matrix_0, a_matrix_1, expected_value, variances, a_matrix_0_model2=0, a_matrix_1_model2=0, compare=False):

    if compare == False:
        for i in range(len(variances)):
            plt.plot(a_matrix_0[i,:], label=r"Var[$p_i$]={}".format(variances[i]))
        plt.xlabel("Individuals")
        plt.ylabel(r"$a_i(x)$")
        plt.title(r"Averaged values of $a_i(x)$ for $x=0$ and $E[p_i]={}$".format(expected_value))
        plt.legend()
        plt.show()

        for i in range(len(variances)):
            plt.plot(a_matrix_1[i,:], label=r"Var[$p_i$]={}".format(variances[i]))
        plt.xlabel("Individuals")
        plt.ylabel(r"$a_i(x)$")
        plt.title(r"Averaged values of $a_i(x)$ for $x=1$ and $E[p_i]={}$".format(expected_value))
        plt.legend()
        plt.show()
    if compare == True:
        fig, ax=plt.subplots(nrows=1, ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].semilogx(a_matrix_0[i, :], label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])), alpha=0.7)
            ax[1].semilogx(a_matrix_0_model2[i, :], label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])), alpha=0.7)
        ax[0].set_xlabel("Individuals")
        ax[1].set_xlabel("Individuals")
        ax[0].set_ylabel(r"$a_i(x)$")
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        ax[0].legend()
        ax[1].legend()
        plt.suptitle(r"Averaged values of $a_i(x)$ for $x=0$ and $E[p_i]={}$".format(expected_value))
        plt.show()
        fig, ax=plt.subplots(nrows=1, ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].plot(a_matrix_1[i, :], label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])), alpha=0.7)
            ax[1].plot(a_matrix_1_model2[i, :], label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])), alpha=0.7)
        ax[0].set_xlabel("Individuals")
        ax[1].set_xlabel("Individuals")
        ax[0].set_ylabel(r"$a_i(x)$")
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        ax[0].legend()
        ax[1].legend()
        plt.suptitle(r"Averaged values of $a_i(x)$ for $x=1$ and $E[p_i]={}$".format(expected_value))
        plt.show()


def correct_cascade(last_cascade_correct, first_cascade_correct, total_fraction_wrong, expected_values,variances, last_cascade_correct_model2=0, first_cascade_correct_model2=0, total_fraction_wrong_mode2=0, compare=False):
    if compare != True:
        for i in range(len(variances)):
            plt.plot(expected_values, last_cascade_correct[i, :], label=r"Var[$p_i$]={}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel("Correct cascades")
        plt.title("Fraction of correct last cascades")
        plt.legend()
        plt.show()
        for i in range(len(variances)):
            plt.plot(expected_values, first_cascade_correct[i,:], label=r"Var[$p_i$]={}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel("Correct cascades")
        plt.title("Fraction of times first cascade is correct")
        plt.legend()
        plt.show()
        for i in range(len(variances)):
            plt.plot(expected_values, total_fraction_wrong[i,:], label=r"Var[$p_i$]={}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel("Wrong decisions")
        plt.title("Fraction of wrong decisions")
        plt.legend()
        plt.show()
    if compare == True:
        fig, ax = plt.subplots(nrows=1,ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].plot(expected_values, last_cascade_correct[i, :], marker = ".",alpha=0.8, markersize=13, linewidth=3,label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expected_values, last_cascade_correct_model2[i, :],
                       label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])),marker = ".", alpha=0.8, markersize=13, linewidth=3)
        ax[0].set_xlabel(r"E[$p_i$]", fontsize=20)
        ax[1].set_xlabel(r"E[$p_i$]", fontsize=20)
        ax[0].set_ylabel(r"Fraction", fontsize=20)
        ax[0].set_title("Model 1", fontsize=20)
        ax[1].set_title("Model 2", fontsize=20)
        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].legend(fontsize=15, loc="lower left")
        ax[1].legend(fontsize=15,loc="lower left")

        plt.tight_layout()
        plt.suptitle("Average number of times the last decision is correct",fontsize=25)
        plt.show()
        fig, ax = plt.subplots(nrows=1,ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].plot(expected_values, first_cascade_correct[i, :], marker = ".",alpha=0.8, markersize=13, linewidth=3, label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expected_values, first_cascade_correct_model2[i, :],
                       marker = ".",alpha=0.8, markersize=13, linewidth=3, label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])))
        ax[0].set_xlabel(r"E[$p_i$]", fontsize=20)
        ax[1].set_xlabel(r"E[$p_i$]", fontsize=20)
        ax[0].set_ylabel(r"Fraction", fontsize=20)
        ax[0].set_title("Model 1", fontsize=20)
        ax[1].set_title("Model 2", fontsize=20)
        ax[0].legend(fontsize=15, loc="lower left")
        ax[1].legend(fontsize=15, loc="lower left")
        ax[0].grid(True)
        ax[1].grid(True)

        plt.tight_layout()
        plt.suptitle("Average number of times the first cascade is correct",fontsize=25)
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].plot(expected_values, total_fraction_wrong[i, :],
                       label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expected_values, total_fraction_wrong_mode2[i, :],
                       label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])))
        ax[0].set_xlabel(r"E[$p_i$]")
        ax[1].set_xlabel(r"E[$p_i$]")
        ax[0].set_ylabel(r"Wrong decisions")
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        ax[0].legend(fontsize=15)
        ax[1].legend(fontsize=15)

        plt.tight_layout()
        plt.suptitle("Average number of wrong decisions", fontsize=15)
        plt.show()


def a_i_fixed_variance(a_matrix_0, a_matrix_1, expected_values, variance, a_matrix_0_model2, a_matrix_1_model2, compare=False):

    if compare == False:
        for i in range(len(expected_values)):
            plt.plot(a_matrix_0[i, :], label= r"E[$p_i$]={}".format(expected_values[i]))
        plt.xlabel("Individuals")
        plt.ylabel(r"$a_i(x)$")
        plt.title(r"Averaged values of $a_i(x)$ for x=0 and $SD[p_i] = {:.2f}$".format(np.sqrt(variance)))
        plt.legend()
        plt.show()

        for i in range(len(expected_values)):
            plt.plot(a_matrix_1[i, :], label= r"E[$p_i$]={}".format(expected_values[i]))
        plt.xlabel("Individuals")
        plt.ylabel(r"$a_i(x)$")
        plt.title(r"Averaged values of $a_i(x)$ for x=1 and $SD[p_i] = {:.2f}$".format(np.sqrt(variance)))
        plt.legend()
        plt.show()
    if compare == True:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        for i in range(0, len(expected_values), 3):
            ax[0].plot(a_matrix_0[i, :], label=r"E[$p_i$]={}".format(expected_values[i]), alpha=0.7)
            ax[1].plot(a_matrix_0_model2[i, :], label=r"E[$p_i$]={}".format(expected_values[i]), alpha=0.7)
        ax[0].set_xlabel("Individuals")
        ax[1].set_xlabel("Individuals")
        ax[0].set_ylabel(r"$a_i(x)$")
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        ax[0].legend()
        ax[1].legend()
        plt.suptitle(r"Averaged values of $a_i(x)$ for x=0 and $SD[p_i] = {:.2f}$".format(np.sqrt(variance)))
        plt.show()
        fig, ax=plt.subplots(nrows=1, ncols=2, sharey=True)
        for i in range(0, len(expected_values), 3):
            ax[0].plot(a_matrix_1[i, :], label=r"E[$p_i$]={}".format(expected_values[i]), alpha=0.7)
            ax[1].plot(a_matrix_1_model2[i, :], label=r"E[$p_i$]={}".format(expected_values[i]), alpha=0.7)
        ax[0].set_xlabel("Individuals")
        ax[1].set_xlabel("Individuals")
        ax[0].set_ylabel(r"$a_i(x)$")
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        ax[0].legend()
        ax[1].legend()
        plt.suptitle(r"Averaged values of $a_i(x)$ for x=1 and $SD[p_i] = {:.2f}$".format(np.sqrt(variance)))
        plt.show()



def a_i_fixed_expectation(a_matrix_0, a_matrix_1, expected_value, variances):
    for i in range(len(variances)):
        plt.plot(a_matrix_0[i, :], label=r"Var[$p_i$]={}".format(variances[i]))
    plt.xlabel("Individuals")
    plt.ylabel(r"$a_i(x)$")
    plt.title(r"Averaged values of $a_i(x)$ for x=0 and $E[p_i] = {}$".format(expected_value))
    plt.legend()
    plt.show()

    for i in range(len(variances)):
        plt.plot(a_matrix_1[i, :], label=r"Var[$p_i$]={}".format(variances[i]))
    plt.xlabel("Individuals")
    plt.ylabel(r"$a_i(x)$")
    plt.title(r"Averaged values of $a_i(x)$ for x=1 and $E[p_i] = {}$".format(expected_value))
    plt.legend()
    plt.show()


def number_of_changes(avg, median, expectations, variances, avg_model2, median_model2, compare=False):

    if compare != True:
        for i in range(len(variances)):
            plt.plot(expectations, median[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Number of changes")
        plt.title(r"Median number of changes after first cascade has occured")
        plt.legend()
        plt.show()
        for i in range(len(variances)):
            plt.plot(expectations, avg[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Number of changes")
        plt.title(r"Median number of changes after first cascade has occured")
        plt.legend()
        plt.show()
    if compare == True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].plot(expectations,median[i, :], marker='.', markersize=13, linewidth=3,  label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expectations, median_model2[i, :], marker='.', markersize=13, linewidth=3, label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        ax[0].set_xlabel(r"E[$p_i$]", fontsize=20)
        ax[0].set_ylabel(r"Number of changes", fontsize=20)
        ax[0].set_title(r"Model 1", fontsize=20)
        ax[0].legend(fontsize=15)
        ax[1].set_xlabel(r"E[$p_i$]", fontsize=20)
        ax[1].set_title(r"Model 2", fontsize=20)
        ax[1].legend(fontsize=15)
        plt.suptitle(r"Median number of changes after first cascade has occured", fontsize=25)
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        for i in range(len(variances)):
            ax[0].plot(expectations, avg[i, :], marker='.', markersize=13, linewidth=3, label=r"$SD[p_i] = {:.2f}$".format(np.sqrt(variances[i])))
            ax[1].plot(expectations, avg_model2[i, :], marker='.', markersize=13, linewidth=3,  label=r"$SD[p_i] = {:.2f}$".format(np.sqrt(variances[i])))
        ax[0].set_xlabel(r"$E[p_i]$", fontsize=20)
        ax[0].set_ylabel(r"Number of changes", fontsize=20)
        ax[0].set_title(r"Model 1", fontsize=20)
        ax[0].legend(fontsize=20)
        ax[1].set_xlabel(r"$E[p_i]$", fontsize=20)
        ax[1].set_title(r"Model 2", fontsize=20)
        ax[1].legend(fontsize=20)
        plt.suptitle(r"Average number of changes after first cascade has occured", fontsize=25)
        plt.show()



def fraction_of_one_change(one_change, expectations, variances, one_change_model2, compare=False):
    if compare == False:
        for i in range(len(variances)):
            plt.plot(expectations, one_change[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Fraction of one change")
        plt.title(r"Fraction of simulations with at least one change after the first cascade has occured")
        plt.legend()
        plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    if compare == True:
        for i in range(len(variances)):
            ax[0].plot(expectations, one_change[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expectations, one_change_model2[i, :], label=r"Var[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        ax[0].set_ylabel(r"Fraction of one change")
        ax[0].set_xlabel(r"E[$p_i$]")
        ax[1].set_xlabel(r"E[$p_i$]")
        ax[0].set_title('Model 1')
        ax[1].set_title('Model 2')


        ax[0].legend()
        ax[1].legend()
        plt.suptitle(r"Fraction of simulations with at least one change after the first cascade has occured")
        plt.show()

def wrong_cascades(average, median, expectations, variances, average_model2, median_model2, compare=False):
    if compare == False:
        for i in range(len(variances)):
            plt.plot(expectations, median[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Wrong cascades")
        plt.title(r"Median number of new wrong cascades after a cascade has occured")
        plt.legend()
        plt.show()
        for i in range(len(variances)):
            plt.plot(expectations, average[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Number of changes")
        plt.title(r"Average number of new wrong cascades after a cascade has occured")
        plt.legend()
        plt.show()
    if compare==True:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        for i in range(len(variances)):
            ax[0].plot(expectations, median[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expectations, median_model2[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        ax[0].set_xlabel(r"E[$p_i$]")
        ax[0].set_ylabel(r"Wrong cascades")
        ax[0].legend()
        ax[1].set_xlabel(r"E[$p_i$]")
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        ax[0].legend()
        plt.suptitle(r"Median number of new wrong cascades after a cascade has occured")
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2)
        for i in range(len(variances)):
            ax[0].plot(expectations, average[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expectations, average_model2[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        ax[0].set_xlabel(r"E[$p_i$]")
        ax[0].set_ylabel(r"Wrong cascades")
        ax[0].legend()
        ax[1].set_xlabel(r"E[$p_i$]")
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        ax[0].legend()
        plt.suptitle(r"Average number of new wrong cascades after a cascade has occured")
        plt.show()


def new_cascades(average, median, expectations, variances, average_model2, median_model2, compare=False):
    if compare == False:
        for i in range(len(variances)):
            plt.plot(expectations, median[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Wrong cascades")
        plt.title(r"Median number of new cascades after a cascade has occured")
        plt.legend()
        plt.show()
        for i in range(len(variances)):
            plt.plot(expectations, average[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Number of changes")
        plt.title(r"Average number of new cascades after a cascade has occured")
        plt.legend()
        plt.show()

    if compare == True:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        for i in range(len(variances)):
            ax[0].plot(expectations, median[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
            ax[1].plot(expectations, median_model2[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        ax[0].legend()
        ax[0].set_ylabel(r"Wrong cascades")
        ax[0].set_xlabel(r"E[$p_i$]")
        ax[1].set_xlabel(r"E[$p_i$]")
        ax[1].legend()
        ax[0].set_title("Model 1")
        ax[1].set_title("Model 2")
        plt.suptitle(r"Median number of new cascades after a cascade has occured")
        plt.show()

        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # for i in range(len(variances)):
        #     ax[0].plot(expectations, average[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        #     ax[1].plot(expectations, average_model2[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        # ax[0].legend()
        # ax[0].set_ylabel(r"Wrong cascades")
        # ax[0].set_xlabel(r"E[$p_i$]")
        # ax[1].set_xlabel(r"E[$p_i$]")
        # ax[1].legend()
        # ax[0].set_title("Model 1")
        # ax[1].set_title("Model 2")
        # plt.suptitle(r"Average number of new cascades after a cascade has occured")
        # plt.show()


def singles(average, median, expectations, variances, average_model2, median_model2, compare=False):
    if compare == False:
        for i in range(len(variances)):
            plt.plot(expectations, median[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Wrong cascades")
        plt.title(r"Median number of outlier-changes after a cascade has occured")
        plt.legend()
        plt.show()
        for i in range(len(variances)):
            plt.plot(expectations, average[i, :], label=r"Var[$p_i$] = {}".format(variances[i]))
        plt.xlabel(r"E[$p_i$]")
        plt.ylabel(r"Number of changes")
        plt.title(r"Average number of outlier-changes after a cascade has occured")
        plt.legend()
        plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for i in range(len(variances)):
        ax[0].plot(expectations, median[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        ax[1].plot(expectations, median_model2[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
    ax[0].legend()
    ax[0].set_ylabel(r"Different decisions")
    ax[0].set_xlabel(r"E[$p_i$]")
    ax[1].set_xlabel(r"E[$p_i$]")
    ax[1].legend()
    ax[0].set_title("Model 1")
    ax[1].set_title("Model 2")
    plt.suptitle(r"Median number of outlier-changes after a cascade has occured")
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    for i in range(len(variances)):
        ax[0].plot(expectations, average[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
        ax[1].plot(expectations, average_model2[i, :], label=r"SD[$p_i$] = {:.2f}".format(np.sqrt(variances[i])))
    ax[0].legend()
    ax[0].set_ylabel(r"Different decisions")
    ax[0].set_xlabel(r"E[$p_i$]")
    ax[1].set_xlabel(r"E[$p_i$]")
    ax[1].legend()
    ax[0].set_title("Model 1")
    ax[1].set_title("Model 2")
    plt.suptitle(r"Average number of outlier-changes after a cascade has occured")
    plt.show()


def distribution(dist, variances, expectation, dist_model2=0, cumulative=False, histtype="bar", compare=False, matrix_plot=True):

    if compare == False:
        for i in range(len(variances)):
            plt.hist(dist[i, :], bins=60, cumulative=cumulative, density=True, histtype=histtype, label=r"Var[$p_i$]={}".format(variances[i]))
            plt.title(r"Distribution of number of individuals until the last cascade occurs for E[$p_i$]={}".format(expectation))
            plt.legend()
            if not cumulative:
                plt.show()
        if cumulative:
            plt.show()
    else:
        if cumulative:
            fig, ax = plt.subplots(nrows=1, ncols=2)
        for i in range(len(variances)):
            if not cumulative:
                fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
            if (cumulative == True) and (histtype == 'bar'):
                ax[0].hist(dist[i, :], bins=60, cumulative=cumulative, density=True, histtype=histtype,
                           label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])), alpha=0.5)
                ax[1].hist(dist_model2[i, :], bins=60, cumulative=cumulative, density=True, histtype=histtype,
                           label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])), alpha=0.5)
                ax[0].legend(loc='lower right')

                ax[1].legend(loc='lower right')
            else:
                ax[0].hist(dist[i, :], bins=60, cumulative=cumulative, density=True, histtype=histtype, label = r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])))
                ax[1].hist(dist_model2[i, :], bins=60, cumulative=cumulative, density=True, histtype=histtype,
                       label=r"SD[$p_i$]={:.2f}".format(np.sqrt(variances[i])))
                if cumulative == True:
                    ax[0].legend(loc='lower right')
                    ax[1].legend(loc='lower right')
                else:
                    ax[0].legend()
                    ax[1].legend()
            ax[0].set_title("Model 1")
            ax[1].set_title("Model 2")
            plt.suptitle(r"Distribution of number of individuals until the last cascade occurs for E[$p_i$]={}".format(expectation))
            if not cumulative:
                plt.show()

        if cumulative:
            plt.show()
    if matrix_plot == True:
        fig, ax = plt.subplots(nrows=3, ncols=2, sharey=True, sharex=True)
        for i in range(len(variances)):
            ax[i, 0].hist(dist[i, :], bins=60, cumulative=cumulative, density=True, histtype=histtype,
                       label=r"$SD[p_i]={:.2f}$".format(np.sqrt(variances[i])))
            ax[i, 1].hist(dist_model2[i, :], bins=60, cumulative=cumulative, density=True, histtype=histtype,
                       label=r"$SD[p_i]={:.2f}$".format(np.sqrt(variances[i])))
            ax[i, 0].axvline(x=np.mean(dist[i, :]), color='r', linestyle='dashed')
            ax[i, 1].axvline(x=np.mean(dist[i, :]), color='r', linestyle='dashed')
            ax[i, 0].legend(fontsize=15)
            ax[i, 1].legend(fontsize=15)
        ax[0, 0].set_title("Model 1", fontsize=18)
        ax[0, 1].set_title("Model 2", fontsize=18)
        plt.suptitle(
            r"Number of individuals until cascade for $E[p_i]={}$".format(expectation), fontsize=21)
        plt.tight_layout()
        # plt.xlabel("Individuals", fontsize=15)
        fig.text(0.5, 0.01, r'Individuals', va='center', ha='center', fontsize=15)
        fig.text(0.01, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=15)

        plt.show()
    return

def numerical_setup(variances, expectations, alpha_values, beta_values):

    x = np.linspace(0, 1.3, 500)
    # for i in range(len(expectations)):
    #     plt.plot(x, beta.pdf(x, alpha_values[i, 0], beta_values[i, 0]), label=r"E[$p_i$]={}".format((beta.mean(alpha_values[i, 0], beta_values[i, 0]))))
    #     plt.show()
    fig, ax = plt.subplots(nrows=len(variances), ncols=5, sharex='row')
    for i in range(len(variances)):
        exp_ind = 0
        var = np.sqrt(variances[i])
        print(var)
        ax[i, exp_ind].set_ylabel(r"$SD[p_i]={:0.2f}$".format(var, fontsize=15))
        # ax[i, exp_ind].set_ytickslabels(fontsize=15)
        ax[i, exp_ind].yaxis.label.set_size(15)
        for j in range(1, len(expectations), 2):

            mean=expectations[j]
            ax[i,exp_ind].plot(x, beta.pdf(x, alpha_values[j, i], beta_values[j, i]))
            if i==0:
                ax[i, exp_ind].set_title(r"$E[p_i]={}$".format(expectations[j]), fontsize=15)
            ax[i, exp_ind].grid(True)
            if i != 2:
                # ax[i, exp_ind].tick_params(axis='x', which='both', bottom = 'off', top="off", labelbottom='off')
                ax[i, exp_ind].xaxis.set_ticklabels([])
                ax[i, exp_ind].xaxis.set_ticks_position('none')

            exp_ind += 1

    plt.tight_layout()
    plt.suptitle("Beta distribution for various parameters", fontsize=17)
    fig.text(0.5, 0.04, r'$p_i$', va='center', ha='center', fontsize=15)
    plt.show()