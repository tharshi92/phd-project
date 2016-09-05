# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 23:05:46 2016

@author: tharshi
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.load('data.npy')
xt = np.load('data_test.npy')
y = np.load('target.npy')
yt = np.load('target_test.npy')

z = np.hstack((x, y))
zt = np.hstack((xt, yt))

d = zt[:, 1:]/z[:, 1:] - 1
t = np.linspace(0, 365, len(d))
titles = ['uwind', 'vwind', 'pressure', 'temperature', 'humidity', 'pbl heights', 'source', 'field']
for i in range(len(d.T)):
    f = plt.figure()
    plt.plot(t, d[:, i])
    plt.title('differences for {0} data'.format(titles[i]))
    plt.xlabel('Date')
    plt.ylabel('% Difference Between 2007 and 2006')
    #plt.savefig('diff_{0}.pdf'.format(titles[i]))