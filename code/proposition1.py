import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from math import floor
import tqdm
from sklearn.linear_model import LinearRegression


def proposition1(option=1):
	"""
	tester plein de valeurs de p puis moyenner les K_n ?
	:return:
	"""
	nb_iters_for_p = int(5e3)
	ls_p = np.random.uniform(size=nb_iters_for_p)

	sample_sizes = np.array(1.5 ** np.arange(5, 25, 1)).astype(int)

	K_n = np.zeros((nb_iters_for_p, len(sample_sizes)))
	résidus = np.zeros((nb_iters_for_p, len(sample_sizes)))
	for i in tqdm.trange(nb_iters_for_p):
		sample = np.random.geometric(ls_p[i],
									 size=sample_sizes[-1])
		set_values = set(sample[0:sample_sizes[0]])
		K_n[i, 0] = len(set_values)
		for j in range(1, len(sample_sizes)):
			set_values |= set(sample[sample_sizes[j - 1]: sample_sizes[j]])
			K_n[i, j] = len(set_values)

		if option != 1:
			résidus[i] = (
				# np.abs(
					(K_n[i] - 1 / 2 * np.log(sample_sizes) ** 2) / np.log(sample_sizes)
				# )
			)
	print(f"K_n=\n{K_n}")

	avg_K_n = np.mean(K_n, axis=0)

	if option == 1:

		reg = LinearRegression().fit(np.log(np.log(sample_sizes)).reshape(-1, 1),
									 np.log(avg_K_n))

		m = reg.coef_[0]
		b = reg.intercept_

		plt.plot(np.log(np.log(sample_sizes)),
				 np.log(avg_K_n),
				 ".-",
				 label="$K_n$ moyen",
				 alpha=.9
				 )
		plt.plot(np.log(np.log(sample_sizes)),
				 m * np.log(np.log(sample_sizes)) + b,
				 label=f"${m:.2f}x$ (pente attendue : 2)",
				 alpha=.7
				 )
		plt.xlabel("$log(log(n))$")
		plt.ylabel("$log(K_n)$")
		plt.title(f"$K_n$ moyenné pour {nb_iters_for_p} valeurs de p  ")
		plt.legend()
		plt.show()
	elif option == 2:
		plt.xscale("log")

		avg_résidus = np.mean(résidus, axis=0)
		std_résidus = np.std(résidus, axis=0)

		plt.title(f"résidus pour la prop 1")
		plt.axhline(color="orange",
					alpha=.7)
		plt.plot(sample_sizes,
				 avg_résidus,
				 ".-",
				 label=f"résidus (moyennés sur {nb_iters_for_p} itérations)",
				 alpha=.9
				 )
		plt.fill_between(sample_sizes,
						 avg_résidus - std_résidus,
						 avg_résidus + std_résidus,
						 alpha=0.3)

		plt.legend()
		plt.show()
	else:
		# spaghetti plots
		nb_spaghetti = 10
		colors = matplotlib.cm.rainbow(np.linspace(0, 1, nb_spaghetti))
		for i in range(nb_spaghetti):
			k = floor(np.random.uniform()*nb_iters_for_p)
			plt.plot(sample_sizes,
					 K_n[k, :],
					 ".-",
					 label=f"trajectoire pour $p={ls_p[k]:.3f}$",
					 color=colors[i],
					 alpha=.8
					 )
		plt.legend(fontsize=8)
		plt.show()


proposition1(option=2)
