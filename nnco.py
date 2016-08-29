# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from init_plotting import *
from nami import Network

def movingAverage(x, window):
    cumsum_vec = np.cumsum(np.insert(x, 0, 0)) 
    ma = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
    return ma
    
x_test = np.load('xt.npy')
y_test = np.load('yt.npy')

# Network Parameters
N = len(x_test)
layers = [len(x_test.T), 8, 4, 2, len(y_test.T)]
reg = 5e-3

# Create Network and Trainer Instances
net = Network(layers, N, reg)
wbs = np.load('weights.npy')
net.set_params(wbs)



#%%

window = 24*14

yt = y_test
zt = net.forward(x_test)
r = zt - yt

yt_ma = movingAverage(yt, window)
zt_ma = movingAverage(zt, window)
r_ma = movingAverage(r, window)

t = np.arange(0, len(yt))/24
t_ma = np.arange(0, len(yt_ma))/24

err = np.linalg.norm((zt - yt)**2)/len(yt)
err_ma = np.linalg.norm((zt_ma - yt_ma)**2)/len(yt_ma)
std = np.std(zt - yt, ddof=1)

init_plotting()
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'#, Smoothing Window = {} hrs'.format(window)
ax.set_xlabel('Days Since Jan 1st 2007')
ax.set_ylabel(ylabel)
reg = 1e-3
ax.set_title(title.format(err, std, layers, reg))
ax.plot(t, yt, label='testing data')
#ax.plot(t_ma, yt_ma, label='testing data (smoothed)')
ax.plot(t, zt, label='network estimate')
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