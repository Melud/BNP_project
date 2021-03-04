import numpy as np
import matplotlib.pyplot as plt
from math import floor
import tqdm
from sklearn.linear_model import LinearRegression


def example1(p, nb_iters=20, option=1):
	"""
	tester une  valeur de p et moyenner
	:return:
	"""
	sample_sizes = np.array(1.5 ** np.arange(5, 35, 1)).astype(int)
	print(sample_sizes)
	K_n = np.zeros((nb_iters, len(sample_sizes)))
	résidus = np.zeros((nb_iters, len(sample_sizes)))
	for i in tqdm.trange(nb_iters):
		sample = np.random.geometric(p,
									 size=sample_sizes[-1])

		set_values = set(sample[0:sample_sizes[0]])
		K_n[i, 0] = len(set_values)
		for j in range(1, len(sample_sizes)):
			set_values |= set(sample[sample_sizes[j - 1]: sample_sizes[j]])
			K_n[i, j] = len(set_values)

		if option != 1:
			résidus[i] = (
			# np.abs(
			K_n[i] -
				(
						np.floor(
							np.log(sample_sizes * p) / np.abs(np.log(1 - p))
						)
						+ 1 + np.euler_gamma / np.abs(np.log(1 - p))
				)
			# )
			)

	# K_n = np.delete(K_n, (0), axis=0)
	print(f"K_n=\n{K_n}")
	if option == 1:
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
	else:
		plt.xscale("log")

		avg_résidus = np.mean(résidus, axis=0)
		std_résidus = np.std(résidus, axis=0)

		plt.title(f"résidus pour $p={p}$")
		plt.plot(sample_sizes,
				 avg_résidus,
				 ".-",
				 label=f"résidus (moyennés sur {nb_iters} itérations)",
				 alpha=.9
				 )
		plt.fill_between(sample_sizes,
						 avg_résidus - std_résidus,
						 avg_résidus + std_résidus,
						 alpha=0.3)

		plt.legend()
		plt.show()


# else:


example1(.001,#np.random.uniform(),
		 nb_iters=100,
		 option=2)
