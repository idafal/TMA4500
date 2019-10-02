import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 500)

alpha = 6.5
b = 6

print("Mean = {}".format(beta.mean(alpha, b)))
print("Var = {}".format(beta.var(alpha, b)))
plt.plot(x, beta.pdf(x, alpha, b))
plt.show()
