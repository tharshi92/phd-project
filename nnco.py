# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from nami import Network, Trainer
from init_plotting import *

def movingAverage(x, window):
    cumsum_vec = np.cumsum(np.insert(x, 0, 0)) 
    ma = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
    return ma

x = np.load('x.npy')
x_test = np.load('xt.npy')
y = np.load('y.npy')
y_test = np.load('yt.npy')
scale_params = np.load('scale_params.npy')

N = len(x)
layers = [len(x.T), 8, len(y.T)]
reg = 9e-3
title_prefix = str(layers)
method = 'BFGS'
net = Network(layers, N, reg, io=True)

#%% Training

trainer = Trainer(net)
trainer.train(x, y, x_test, y_test, method=method)
wbs = net.get_params()
np.save('weights_{}'.format(layers), wbs)
#%%
plt.rcParams['text.usetex'] = True
init_plotting()
ax = plt.subplot(111)
f_c = plt.gcf()
plt.margins(0.0, 0.1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training History, Layers = {}, Reg={:.4e}, Method = {}'.format(layers, reg, 'BFGS'))
ax.loglog(trainer.J, label='Training')
ax.loglog(trainer.J_test, label='Testing')
plt.legend()
savename = 'costsnnco.pdf'
f_c.set_size_inches(width, height)
plt.savefig(savename, bbox_inches='tight')

#%% Fit on Test Data
width  = 7.784
height = width / 1.618

window = 24*7

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

plt.rcParams['text.usetex'] = True
init_plotting()
ax = plt.subplot(111)
f = plt.gcf()
ylabel = 'CO Field (ppbv)'#, Smoothing Window = {} hrs'.format(window)
title1 = 'Neural Network Fit, MSE = {:e}, $\sigma$ = {:e}'
title2 = ', Layers: {}, Reg = {}'
title = title1 + title2
ax.set_xlabel('Days Since Jan 1st 2007')
ax.set_ylabel(ylabel)
reg = 1e-3
ax.set_title(title.format(err, std, layers, reg))
ax.plot(t, yt, label='testing data')
#ax.plot(t_ma, yt_ma, label='testing data (smoothed)')
ax.plot(t, zt, label='network estimate')
ax.plot(t, r, label='residuals')
plt.legend(loc='center left')
f.set_size_inches(width, height)
savename = 'nnfield_{}_{}_EXTRA'.format(window, layers) + method + '.pdf'
plt.savefig(savename, bbox_inches='tight')

#%% Plot Residual Analysis

iqr = np.subtract(*np.percentile(r, [75, 25]))
h = 2*iqr*len(r)**(-1/3)
b  = (r.max() - r.min())/h

init_plotting()
ax = plt.subplot(111)  
f2 = plt.gcf()
title1 = 'Residual Distribution, MSE = {:e}, $\sigma$ = {:e}'
title2 = ', Layers: {}, Reg = {}'
title = title1 + title2
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title(title.format(err, std, layers, reg))
hist = plt.hist(r, bins=int(b))
f2.set_size_inches(width, height)
savename = 'residuals_{}_{}'.format(window, layers) + method + '.pdf'
plt.savefig(savename, bbox_inches='tight')