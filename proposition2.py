# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:25:53 2021

@author: DORON
"""
import numpy as np 
import matplotlib.pyplot as plt


def prop2(n_sample_for_p,n_power_max,n_weigths):
    p=np.random.rand(n_sample_for_p).reshape(-1,1)
    range_sizes=np.array(1.5**np.arange(3,n_power_max)).astype(int)
    K_n=np.zeros((n_sample_for_p,range_sizes.shape[0]))
    n=np.arange(1,n_weigths+1).reshape(1,-1)
    weights=p*(1-p)**(n-1) *((1+n*p)/2)
    for i in range(n_sample_for_p):
        prob=weights[i,:]
     
        list_cluster=np.random.choice(n.reshape(-1,),range_sizes[-1],p=prob/prob.sum())
        K_n[i,]=[np.unique(list_cluster[:int(j)]).shape[0] for j in range_sizes]
    return K_n
            
        
    

def theorical_asymp(n_power_max):
    range_sizes=np.array(1.5**np.arange(3,n_power_max)).astype(int)
    return(0.5*np.log(range_sizes)**2 +np.log(range_sizes)*np.log(np.log(range_sizes)) -(1+np.log(2))*np.log(range_sizes))



K_n=prop2(100,30,10**5)
K_n=K_n.mean(axis=0)
range_sizes=np.array(1.5**np.arange(3,30)).astype(int)
plt.figure()
plt.semilogy(range_sizes,K_n,ls='--',label="K_n")
plt.semilogy(range_sizes,theorical_asymp(30) ,ls='--',label="Théorie")
plt.legend()
plt.show()




