import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.linear_model import LinearRegression

def example1(p):
	"""
	tester une  valeur de p et moyenner
	:return:
	"""
	nb_iters_for_p = int(20)
	sample_sizes = np.array(1.5 ** np.arange(5, 35, 3)).astype(int)
	K_n = np.zeros(len(sample_sizes))
	for _ in tqdm.trange(nb_iters_for_p):
		n_iters_for_weigths = int(1e3)
		w = p * (1 - p) ** (np.arange(0, n_iters_for_weigths))  # np.zeros(n_iters_for_weigths)

		x = []
		y = []

		sample = np.random.choice(range(n_iters_for_weigths), size=sample_sizes[-1], p=w / (np.sum(w)))
		for size in sample_sizes:
			sub_sample = sample[0:size]
			x.append(len(sub_sample))
			y.append(len(np.unique(sub_sample)))
		K_n = np.vstack(
			(K_n,
			 y)
		)

	K_n = np.delete(K_n, (0), axis=0)
	print(f"K_n=\n{K_n}")
	avg_K_n = np.mean(K_n, axis=0)

	reg = LinearRegression().fit(np.log(sample_sizes).reshape(len(sample_sizes),1),
	                             (avg_K_n))
	m = reg.coef_[0]
	b = reg.intercept_

	plt.xscale("log")

	plt.plot(sample_sizes,
	         avg_K_n,
	         ".-",
	         label="K_n moyen",
	         alpha=.9
	         )
	plt.plot(sample_sizes,
	         m*np.log(sample_sizes)+b,
	         label=f"{m:.3f}*x (predicted {abs(np.log(1-p))**-1:.3f})",
	         alpha=.7
	         )
	plt.xlabel("$log(n)$")
	plt.ylabel("$K_n$")
	plt.legend()
	plt.show()

example1(.25)