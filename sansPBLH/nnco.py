# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from init_plotting import *
from nami import Network
from config import nn, reg
    
x_test = np.load('xt.npy')
y_test = np.load('yt.npy')

# Network Parameters
N = len(x_test)
layers = [len(x_test.T), nn, len(y_test.T)]

# Create Network and Trainer Instances
net = Network(layers, N, reg)
wbs = np.load('weights.npy')
net.set_params(wbs)

#%%

y = y_test
z = net.forward(x_test)
r = z - y

t = np.arange(0, len(y))/24

err = np.linalg.norm(r**2)/len(r)
std = np.std(r, ddof=1)

f0 = plt.figure()
init_plotting()
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'
ax.set_xlabel('Days Since Jan 1st 2007')
ax.set_ylabel(ylabel)
reg = 1e-3
ax.set_title('Fit')
ax.plot(t, y, label='testing data')
ax.plot(t, z, label='network estimate')
ax.plot(t, r, label='residuals')
plt.legend(loc='center left')
f0.savefig('fit_2.pdf', bbox_inches='tight')
plt.show()


iqr = np.subtract(*np.percentile(r, [75, 25]))
h = 2*iqr*len(r)**(-1/3)
b  = (r.max() - r.min())/h

f = plt.figure()
init_plotting()
ax = plt.subplot(111)  
title = 'MSE = {:e}, $\sigma$ = {:e}'
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title(title.format(err, std, layers, reg))
hist = plt.hist(r, bins=int(b), alpha=0.5)
f.savefig('hist_2.pdf', bbox_inches='tight')
plt.show()
