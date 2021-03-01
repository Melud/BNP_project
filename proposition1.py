import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.linear_model import LinearRegression

def proposition1():
	"""
	tester plein de valeurs de p puis moyenner les K_n ?
	:return:
	"""
	nb_iters_for_p = int(50)
	sample_sizes = np.array(1.5 ** np.arange(5, 35, 1)).astype(int)
	K_n = np.zeros(len(sample_sizes))
	for _ in tqdm.trange(nb_iters_for_p):
		p = np.random.uniform()

		n_iters_for_weigths = int(1e3)
		# w = p * (1 - p) ** (np.arange(0, n_iters_for_weigths))

		x = []
		y = []

		sample = np.random.geometric(p,
		                             size=sample_sizes[-1])
		#np.random.choice(range(n_iters_for_weigths), size=sample_sizes[-1], p=w / (np.sum(w)))
		for size in sample_sizes:
			sub_sample = sample[0:size]
			x.append(len(sub_sample))
			y.append(len(np.unique(sub_sample)))
		K_n = np.vstack(
			(K_n,
			 np.array(y))
			## TEMP : p*y au lieu de y
		)

	K_n = np.delete(K_n, (0), axis=0)
	print(f"K_n=\n{K_n}")
	avg_K_n = np.mean(K_n, axis=0)

	# reg = LinearRegression().fit(np.log(sample_sizes).reshape(len(sample_sizes),1),
	#                              (avg_K_n))
	# m = reg.coef_[0]
	# b = reg.intercept_

	plt.xscale("log")

	plt.plot(sample_sizes,
	         avg_K_n,
	         ".-",
	         label="K_n moyen",
	         alpha=.9
	         )
	# plt.plot(sample_sizes,
	#          m*np.log(sample_sizes)+b,
	#          # label=f"{m:.3f}*x (predicted {abs(np.log(1-p))**-1:.3f})",
	#          alpha=.7
	#          )
	plt.xlabel("$log(n)$")
	plt.ylabel("$K_n$")
	plt.legend()
	plt.show()

# unique, counts = np.unique(sample, return_counts=True)

# print(dict(zip(unique, counts)))

proposition1()