# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from init_plotting import *
from nami import Network
    
x_test = np.load('xt.npy')
y_test = np.load('yt.npy')

# Network Parameters
N = len(x_test)
layers = [len(x_test.T), 8, len(y_test.T)]
reg = 5e-3

# Create Network and Trainer Instances
net = Network(layers, N, reg)
wbs = np.load('weights.npy')
net.set_params(wbs)

#%%



y = y_test
z = net.forward(x_test)
r = z - y

t = np.arange(0, len(yt))/24

err = np.linalg.norm(r**2)/len(r)
std = np.std(r, ddof=1)

init_plotting()
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'#, Smoothing Window = {} hrs'.format(window)
ax.set_xlabel('Days Since Jan 1st 2007')
ax.set_ylabel(ylabel)
reg = 1e-3
ax.set_title('Fit')
ax.plot(t, y, label='testing data')
ax.plot(t, z, label='network estimate')
ax.plot(t, r, label='residuals')
plt.legend(loc='center left')
plt.show()


iqr = np.subtract(*np.percentile(r, [75, 25]))
h = 2*iqr*len(r)**(-1/3)
b  = (r.max() - r.min())/h

plt.clf()
init_plotting()
ax = plt.subplot(111)  
title = 'MSE = {:e}, $\sigma$ = {:e}'
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title(title.format(err, std, layers, reg))
hist = plt.hist(r, bins=int(b), alpha=0.5)
plt.show()
