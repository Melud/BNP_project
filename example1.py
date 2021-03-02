import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.linear_model import LinearRegression


def example1(p):
	"""
	tester une  valeur de p et moyenner
	:return:
	"""
	nb_iters = int(20)
	sample_sizes = np.array(1.5 ** np.arange(5, 35, 1)).astype(int)
	K_n = np.zeros(len(sample_sizes))
	for _ in tqdm.trange(nb_iters):
		# n_iters_for_weigths = int(1e3)

		x = []
		y = []

		sample = np.random.geometric(p,
		                             size=sample_sizes[-1])
		for size in sample_sizes:
			x.append(size)
			y.append(len(np.unique(sample[0:size])))
		K_n = np.vstack(
			(K_n,
			 y)
		)

	K_n = np.delete(K_n, (0), axis=0)
	print(f"K_n=\n{K_n}")
	avg_K_n = np.mean(K_n, axis=0)

	reg = LinearRegression().fit(np.log(sample_sizes).reshape(len(sample_sizes), 1),
	                             (avg_K_n))
	m = reg.coef_[0]
	b = reg.intercept_

	plt.xscale("log")

	plt.plot(sample_sizes,
	         avg_K_n,
	         ".-",
	         label=f"$K_n$ (moyenné sur {nb_iters} itérations)",
	         alpha=.9
	         )
	plt.plot(sample_sizes,
	         m * np.log(sample_sizes) + b,
	         label=f"${m:.2f}x$ (pente attendue : 1/|log(1-p)|≈{abs(np.log(1 - p)) ** -1:.2f})",
	         alpha=.7
	         )
	plt.xlabel("$n$")
	plt.ylabel("$K_n$")
	plt.title(f"$K_n$ avec $p={p:.3f}$")
	plt.legend()
	plt.show()


example1(np.random.uniform())
