import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.linear_model import LinearRegression


# import cProfile


def nb_unique(a):
	return len(np.unique(a))


nb_unique_v = np.vectorize(nb_unique)


def proposition1():
	"""
	tester plein de valeurs de p puis moyenner les K_n ?
	:return:
	"""
	nb_iters_for_p = int(5e3)
	ls_p = np.random.uniform(size=nb_iters_for_p)

	sample_sizes = np.array(1.5 ** np.arange(12, 25, 1)).astype(int)

	K_n = np.zeros((nb_iters_for_p, len(sample_sizes)))
	for i in tqdm.trange(nb_iters_for_p):
		sample = np.random.geometric(ls_p[i],
		                             size=sample_sizes[-1])
		set_values = set(sample[0:sample_sizes[0]])
		K_n[i, 0] = len(set_values)
		for j in range(1, len(sample_sizes)):
			set_values |= set(sample[sample_sizes[j - 1]: sample_sizes[j]])
			K_n[i, j] = len(set_values)

	print(f"K_n=\n{K_n}")

	avg_K_n = np.mean(K_n, axis=0)

	reg = LinearRegression().fit(np.log(np.log(sample_sizes)).reshape(-1, 1),
	                             np.log(avg_K_n))
	# reg = LinearRegression().fit((np.log(sample_sizes) ** 2).reshape(-1, 1),
	#                              (avg_K_n))
	m = reg.coef_[0]
	b = reg.intercept_

	# plt.xscale("log")

	plt.plot(np.log(np.log(sample_sizes)),
	         np.log(avg_K_n),
	         ".-",
	         label="$K_n$ moyen",
	         alpha=.9
	         )
	plt.plot(np.log(np.log(sample_sizes)),
	         m * np.log(np.log(sample_sizes)) + b,
	         # label=f"{m:.3f}*x (predicted {abs(np.log(1-p))**-1:.3f})",
	         label=f"${m:.2f}x$ (pente attendue : 2)",
	         alpha=.7
	         )
	plt.xlabel("$log(log(n))$")
	plt.ylabel("$log(K_n)$")
	plt.title(f"$K_n$ moyenné pour {nb_iters_for_p} valeurs de p  ")
	plt.legend()
	plt.show()


proposition1()
# proposition1_option2()
