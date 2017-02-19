# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from nami import Network
from helperFunctions import *
from prepNetwork import start, end
from config import *    
import pickle
import sys

localRepo = homeDir + sys.argv[1] + '/'

# change to data directory
os.chdir(localRepo)

x = np.load('x.npy')
y = np.load('y.npy')
sP = pickle.load(open('scaleParams.p', 'rb'))

# Network Parameters
N = len(x)
layers = [len(x.T), hiddenNeurons, len(y.T)]

# change to specific run directory
os.chdir(saveDir + sys.argv[1] + '/nami')

# Create Network and Trainer Instances
net = Network(layers, N, reg)
wbs = np.load('weights.npy')
net.set_params(wbs)

yp = sP[2] *  y
z = sP[2] * net.forward(x)
r = z - yp
t = np.arange(0, len(yp))/24

y_ty = yp[start:end, :]
z_ty = z[start:end, :]
r_ty = r[start:end, :]
t_ty = np.arange(0, len(y_ty))/24

err = np.linalg.norm(r_ty**2)/len(r_ty)
m = np.float(np.mean(r_ty, axis=0))
std = np.std(r_ty, ddof=1)

# save datafiles
np.save('pred', z)

f0 = plt.figure(figsize=(12, 8))
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'
ax.set_xlabel('Days Since Jan 1st 2006')
ax.set_ylabel(ylabel)
ax.set_title('Fit')
ax.plot(t, yp, label='testing data')
ax.plot(t, z, label='network estimate')
ax.plot(t, r, label='residuals')
plt.legend(loc='center left')
f0.savefig('fit.png', bbox_inches='tight')

f1 = plt.figure(figsize=(12, 8))
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'
ax.set_xlabel('Days Since Jan 1st ' + str(testingYear))
ax.set_ylabel(ylabel)
ax.set_title('Fit (Testing Year Only)')
ax.plot(t_ty, y_ty, label='testing data')
ax.plot(t_ty, z_ty, label='network estimate')
ax.plot(t_ty, r_ty, label='residuals')
plt.legend(loc='center left')
f1.savefig('fit_testyear.png', bbox_inches='tight')

f2 = plt.figure()
ax = plt.subplot(111)  
title = 'MSE = {:3f}, $\sigma$ = {:3f}, $\mu$ = {:3f}'
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title(title.format(err, std, m))
hist = plt.hist(r, alpha=0.5)
f2.savefig('hist.png', bbox_inches='tight')

windows = [24, 3 * 24, 7 * 24, 30 * 24, 120 * 24]

for w in windows:
	zw = moving_average(z_ty, w)
	yw = moving_average(y_ty, w)
	rw = moving_average(r_ty, w)
	tw = moving_average(t_ty, w)/(365./12)
	f = plt.figure(figsize=(12, 8))
	ax = plt.subplot(111)
	ylabel = 'CO Field (Moving Average {0} Days, ppbv)'.format(w/24)
	ax.set_xlabel('Months Since Jan 1st 2007')
	ax.set_ylabel(ylabel)
	ax.set_title('Fit')
	ax.plot(tw, yw, label='testing data', alpha=0.5)
	ax.plot(tw, zw, label='network estimate')
	ax.plot(tw, rw, label='residual')
	plt.legend()
	f.savefig('fit_window_{0}.png'.format(str(w/24)), bbox_inches='tight')



plt.show()
