# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:25:53 2021

@author: DORON
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def prop2(n_sample_for_p, n_power_max, n_weigths):
	p = np.random.rand(n_sample_for_p).reshape(-1, 1)
	range_sizes = np.array(1.5 ** np.arange(15, n_power_max, .5)).astype(int)
	K_n = np.zeros((n_sample_for_p, range_sizes.shape[0]))
	n = np.arange(1, n_weigths + 1).reshape(1, -1)
	weights = p * (1 - p) ** (n - 1) * ((1 + n * p) / 2)
	for i in range(n_sample_for_p):
		prob = weights[i, :]

		list_cluster = np.random.choice(n.reshape(-1, ), range_sizes[-1], p=prob / prob.sum())
		K_n[i,] = [np.unique(list_cluster[:int(j)]).shape[0] for j in range_sizes]
	return K_n, range_sizes


def theorical_asymp(n_power_max):
	range_sizes = np.array(1.5 ** np.arange(3, n_power_max)).astype(int)
	return (0.5 * np.log(range_sizes) ** 2 + np.log(range_sizes) * np.log(np.log(range_sizes)) - (
			1 + np.log(2)) * np.log(range_sizes))


n_sample_for_p = 1000
K_n, range_sizes = prop2(n_sample_for_p, 30, 10 ** 5)
K_n = K_n.mean(axis=0)
# range_sizes = np.array(1.5 ** np.arange(3, 30)).astype(int)
plt.figure()

plt.plot(
	np.log(np.log(range_sizes)),
	np.log(K_n),
	"-.",
	label="$K_n$ moyen"
)

reg = LinearRegression().fit(
	np.log(np.log(range_sizes))[:, None],
	np.log(K_n)
)
m = reg.coef_[0]
b = reg.intercept_

plt.title(f"$K_n$ moyenné pour {n_sample_for_p} valeurs de p  ")
plt.xlabel("$log(log(n))$")
plt.ylabel("$log(K_n)$")

plt.plot(
	np.log(np.log(range_sizes)),
	m * np.log(np.log(range_sizes)) + b,
	label=f"${m:.2f}x$ (predicted $2x$)",
)
# plt.semilogy(range_sizes, K_n, '-.', label="K_n")
# plt.semilogy(range_sizes, theorical_asymp(30), '-.', label="Théorie")
plt.legend()
plt.show()
