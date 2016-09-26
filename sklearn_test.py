# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:39:12 2016

@author: tsri
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

N = 1000
reg = 1e-3

t = np.random.uniform(low=230, high=310, size=(int(N), 1))
t2 = np.random.uniform(low=230, high=310, size=(int(N), 1))
d = np.random.uniform(low=0, high=365, size=(int(N), 1))
d2 = np.random.uniform(low=0, high=365, size=(int(N), 1))

temp1 = np.linspace(0, 365, 1000)
temp2 = np.linspace(200, 350, 1000)
dd, tt = np.meshgrid(temp1, temp2)

def mice(d, t):
    return (50 * np.sin(-2*np.pi*(d + 180)/30) + 100) * \
        np.exp(-(t - 280)**2 /100000.0)
    
truth = mice(dd, tt)
y = np.zeros(int(N)).reshape((N, 1)) + 5*np.random.randn()
y2 = np.zeros(int(N)).reshape((N, 1)) + 5*np.random.randn()

for i in range(N):
    y[i] = mice(d[i], t[i])
    y2[i] = mice(d2[i], t2[i])

x = np.hstack((d, t))
x2 = np.hstack((d2, t2))

mu_x = np.mean(x, axis=0)
s_x = np.std(x, axis=0, ddof=1)
mu_y = np.mean(y, axis=0)
s_y = np.std(y, axis=0, ddof=1)

reg = MLPRegressor(solver='adam')
#reg.fit(X, y)                         
#reg.predict([1., 2.])
#reg.predict([0., 0.])