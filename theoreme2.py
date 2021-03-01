import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import factorial


def proposition2(m,n_power_max,n_sample_for_p):
    range_sizes=np.array(1.5**np.arange(3,n_power_max)).astype(int)
    K_n=np.zeros((n_sample_for_p,range_sizes.shape[0]))
    p=np.exp(-np.random.gamma(m+1,1,n_sample_for_p)).reshape(-1,1)
    list_cluster=np.random.geometric(p,size=(n_sample_for_p,range_sizes[-1]))
    for i in range(n_sample_for_p):
        K_n[i,]=[np.unique(list_cluster[i,:int(j)]).shape[0] for j in range_sizes]
    return K_n



def theorical_asymp(n_power_max):
    range_sizes=np.array(1.5**np.arange(3,n_power_max)).astype(int)
    return ((np.log(range_sizes)**(m+2))/(factorial(m+2)) + (0.5772156649/factorial(m+1))*np.log(range_sizes)**2)
K_n=proposition2(1,30,100)
K_n=K_n.mean(axis=0)
range_sizes=np.array(1.5**np.arange(3,30)).astype(int)
plt.figure()
plt.loglog(range_sizes,K_n,ls='--',label="K_n")
plt.loglog(range_sizes,((np.log(range_sizes))**3)/6 + (np.log(range_sizes)**2)*(0.5772156649/2) ,ls='--',label="Théorie")
plt.legend()
plt.show()



for m in range(1,10):
    K_n=proposition2(m,30,200).mean(axis=0)
    range_sizes=np.array(1.5**np.arange(3,30)).astype(int)
    plt.figure()
    plt.semilogx(range_sizes,K_n,label=f"K_n pour m={m}")
    plt.semilogx(range_sizes,theorical_asymp(30),label=f"Théorie pour m={m}")
    plt.legend()
    plt.show()
    
          