import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from sklearn.linear_model import LinearRegression


def proposition2(m, n_power_max, n_sample_for_p):
	range_sizes = np.array(1.5 ** np.arange(15, n_power_max)).astype(int)
	K_n = np.zeros((n_sample_for_p, range_sizes.shape[0]))
	p = np.exp(-np.random.gamma(m + 1, 1, n_sample_for_p)).reshape(-1, 1)
	list_cluster = np.random.geometric(p, size=(n_sample_for_p, range_sizes[-1]))
	for i in range(n_sample_for_p):
		K_n[i,] = [np.unique(list_cluster[i, :int(j)]).shape[0] for j in range_sizes]
	return K_n, range_sizes


def theorical_asymp(n_power_max):
	range_sizes = np.array(1.5 ** np.arange(3, n_power_max)).astype(int)
	return ((np.log(range_sizes) ** (a + 2)) / (factorial(a + 2)) + (0.5772156649 / factorial(a + 1)) * np.log(
		range_sizes) ** 2)


m = 5

n_sample_for_p = 1000
K_n, range_sizes = proposition2(m, 30, n_sample_for_p)

K_n = K_n.mean(axis=0)
#
# plt.figure()
# plt.loglog(range_sizes, K_n, ls='--', label="K_n")
# plt.loglog(range_sizes, ((np.log(range_sizes)) ** 3) / 6 + (np.log(range_sizes) ** 2) * (0.5772156649 / 2), ls='--',
#            label="Théorie")
# plt.legend()
# plt.show()

plt.plot(
	np.log(np.log(range_sizes)),
	np.log(K_n),
	".-",
	label="$K_n$ moyen",
	alpha=.9
)

reg = LinearRegression().fit(
	np.log(np.log(range_sizes))[:, None],
	np.log(K_n)
)
a = reg.coef_[0]
b = reg.intercept_

plt.title(f"$K_n$ moyenné pour {n_sample_for_p} valeurs de p  lorsque m={m}")
plt.xlabel("$log(log(n))$")
plt.ylabel("$log(K_n)$")

plt.plot(
	np.log(np.log(range_sizes)),
	a * np.log(np.log(range_sizes)) + b,
	label=f"${a:.2f}x$ (predicted ${m + 2}x$)",
	alpha=.7
)
plt.legend()
plt.show()