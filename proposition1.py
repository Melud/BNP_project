import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.linear_model import LinearRegression


def proposition1():
	"""
	tester plein de valeurs de p puis moyenner les K_n ?
	:return:
	"""
	nb_iters_for_p = int(500)
	ls_p = np.random.uniform(size=nb_iters_for_p)
	sample_sizes = np.array(1.5 ** np.arange(10, 35, 1)).astype(int)
	K_n = np.zeros(len(sample_sizes))
	for i in tqdm.trange(nb_iters_for_p):
		p = ls_p[i]

		x = []
		y = []

		sample = np.random.geometric(p,
		                             size=sample_sizes[-1])
		# np.random.choice(range(n_iters_for_weigths), size=sample_sizes[-1], p=w / (np.sum(w)))
		for size in sample_sizes:
			x.append(size)
			y.append(len(np.unique(sample[0:size])))
		K_n = np.vstack(
			(K_n,
			 np.array(y))
			## TEMP : p*y au lieu de y
		)

	K_n = np.delete(K_n, (0), axis=0)
	print(f"K_n=\n{K_n}")

	print("OPTION 1 FINIE")

	avg_K_n = np.mean(K_n, axis=0)

	reg = LinearRegression().fit(np.log(np.log(sample_sizes)).reshape(-1, 1),
	                             np.log(avg_K_n))
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


def proposition1_option2():
	"""
	tester plein de valeurs de p *autrement*
	 puis moyenner les K_n ?
	:return:
	"""
	sample_sizes = np.array(1.5 ** np.arange(10, 35, 1)).astype(int)
	ls_p = np.random.uniform(size=sample_sizes[-1])

	x = []

	for i in tqdm.trange(sample_sizes[-1]):
		x.append(np.random.geometric(ls_p[i],
		                             size=1))
	K_n = []
	for size in sample_sizes:
		K_n.append(len(np.unique(x[0:size])))
	print(f"K_n=\n{K_n}")

	print("OPTION 2 FINIE")

	reg = LinearRegression().fit(np.log(np.log(sample_sizes)).reshape(-1, 1),
	                             np.log(K_n))
	m = reg.coef_[0]
	b = reg.intercept_

	plt.plot(np.log(np.log(sample_sizes)),
	         np.log(K_n),
	         ".-",
	         label="K_n moyen",
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
	plt.legend()
	plt.show()


proposition1()
# proposition1_option2()
