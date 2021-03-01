import numpy as np
import matplotlib.pyplot as plt
import tqdm

def proposition1():
	"""
	tester plein de valeurs de p puis moyenner les K_n ?
	:return:
	"""
	nb_iters_for_p = int(50)
	range_sizes = np.array(1.5 ** np.arange(5, 35, 3)).astype(int)
	K_n = np.zeros(len(range_sizes))
	for _ in tqdm.trange(nb_iters_for_p):
		p = np.random.uniform()  # 1e-1  # np.random.uniform()
		# print(f"p={p}")
		# size_sample = int(1e6)
		n_iters_for_weigths = int(1e3)
		w = p * (1 - p) ** (np.arange(0, n_iters_for_weigths))  # np.zeros(n_iters_for_weigths)
		# print(np.sum(w))
		# w[0] = p
		# for i in range(1, n_iters_for_weigths):
		# 	w[i] = (1 - p) * w[i - 1]

		# sample = []
		x = []
		y = []

		sample = np.random.choice(range(n_iters_for_weigths), size=range_sizes[-1], p=w / (np.sum(w)))
		for size in range_sizes:
			sub_sample = sample[0:size]
			x.append(len(sub_sample))
			y.append(len(np.unique(sub_sample)))
		K_n = np.vstack(
			(K_n,
			 y)
		)
		# unique =
		# print(len(unique))
		# np.hstack((np.array(range(10)), np.array(range(4))))
		# print(x)
		# print(y)
	K_n = np.delete(K_n, (0), axis=0)
	print(f"K_n=\n{K_n}")
	avg_K_n = np.mean(K_n, axis=0)

	plt.plot(np.log(range_sizes), avg_K_n,".-",

	         )
	plt.xlabel("$log(n)$")
	plt.ylabel("$K_n$")
	plt.show()

# unique, counts = np.unique(sample, return_counts=True)

# print(dict(zip(unique, counts)))

proposition1()