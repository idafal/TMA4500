import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
#
alpha = 1.8
b = 0.2
#
#
x = np.linspace(0, 2, 500)


mean_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
variance_values = [0.002, 0.008, 0.03]

x_alpha, x_beta = symbols("x_alpha x_beta")


def find_alpha_beta(mean_values, variance_values):
    values_alpha = np.zeros((len(mean_values), len(variance_values)))
    values_beta = np.zeros((len(mean_values), len(variance_values)))
    for i in range(len(variance_values)):
        for j in range(len(mean_values)):
            beta_distribution_mean = (x_alpha) / (x_alpha + x_beta) - mean_values[j]
            beta_distribution_variance = (x_alpha * x_beta) / ((x_alpha + x_beta) ** 2 * (x_alpha + x_beta + 1)) - \
                                         variance_values[i]

            result = solve((beta_distribution_mean, beta_distribution_variance), (x_alpha, x_beta))
            alpha = np.float(result[0][0])
            b = np.float(result[0][1])
            values_alpha[j, i] = alpha
            values_beta[j, i] = b
    print(values_alpha)
    print(values_beta)
    return values_alpha, values_beta

values_alpha, values_beta = find_alpha_beta(mean_values, variance_values)
i=2
for j in range(10):
    print("sd = {}: ".format(np.sqrt(variance_values[i])))
    print("sd = {:.2f} \t E[p]= {}\t alpha= {} \t beta={} ".format(np.sqrt(variance_values[i]), mean_values[j],values_alpha[j, i], values_beta[j,i]))
# np.save("variables/alpha_values", values_alpha, allow_pickle=True)
# np.save("variables/beta_values", values_beta, allow_pickle=True)
# np.save("variables/mean_values", mean_values, allow_pickle=True)
# np.save("variables/variance_values", variance_values, allow_pickle=True)

for i in range(len(variance_values)):
    mean = beta.mean(values_alpha[2,i], values_beta[2, i])
    print("Mean = {}".format(mean))
    print("Var = {}".format(beta.var(values_alpha[2,i], values_beta[2, i])))
    plt.plot(x, beta.pdf(x, values_alpha[2,i], values_beta[2, i]), label=r"$\alpha = {}, \beta = {}$".format(values_alpha[2,i], values_beta[2, i]))
    plt.vlines(x=mean, ymin=1, ymax=max(beta.pdf(x, values_alpha[2,i], values_beta[2, i])) + 2, color="red", label="Mean = {}".format(mean))
    # plt.fill_betweenx(x, x2=mean-beta.var(alpha, b), x2=mean + beta.var(alpha, b), facecolor="blue", alpha=2.5)
    plt.title("Beta distribution - pdf")
    plt.legend()
    plt.show()

for i in range(len(mean_values)):
    mean = beta.mean(values_alpha[i,0], values_beta[i, 0])
    print("Mean = {}".format(mean))
    print("Var = {}".format(beta.var(values_alpha[i, 0], values_beta[i, 0])))
    plt.plot(x, beta.pdf(x, values_alpha[i, 0], values_beta[i, 0]), label=r"$\alpha = {}, \beta = {}$".format(values_alpha[i, 0], values_beta[i, 0]))
    plt.vlines(x=mean, ymin=1, ymax=max(beta.pdf(x, values_alpha[i, 0], values_beta[i, 0])) + 1, color="red", label="Mean = {}".format(mean))
    # plt.fill_betweenx(x, x2=mean-beta.var(alpha, b), x2=mean + beta.var(alpha, b), facecolor="blue", alpha=2.5)
    plt.title("Beta distribution - pdf")
    plt.legend()
    plt.grid(True)
    plt.show()