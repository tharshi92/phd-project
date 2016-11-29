from config import *
from helperFunctions import *
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# change to specific run directory
os.chdir(saveDir + sys.argv[1] + '/tf')

z = np.load('z.npy')
y = np.load('signal.npy')
r = z - y

t = np.arange(0, len(y))/24
err = np.linalg.norm(r**2)/len(r)
m = np.float(np.mean(r, axis=0))
std = np.std(r, ddof=1)

f_CO = plt.figure(figsize=(12, 8))
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'
ax.set_xlabel('Days Since Jan 1st 2007')
ax.set_ylabel(ylabel)
ax.set_title('Fit')
ax.plot(t, y, label='testing data')
ax.plot(t, z, label='network estimate')
ax.plot(t, r, label='residuals')
plt.legend(loc='center left')
savename = 'fit_ty.png'
plt.savefig(savename, bbox_inches='tight')

f_hist = plt.figure()
ax = plt.subplot(111)  
title = 'MSE = {:.3e}, $\sigma$ = {:.3e}, $\mu$ = {:.3e}'
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title(title.format(err, std, m))
hist = plt.hist(r, alpha=0.5)
savename = 'hist.png'
plt.savefig(savename, bbox_inches='tight')

windows = [24, 3 * 24, 7 * 24, 30 * 24, 120 * 24]

for w in windows:
	zw = moving_average(z, w)
	yw = moving_average(y, w)
	f = plt.figure(figsize=(12, 8))
	ax = plt.subplot(111)
	ylabel = 'CO Field (Moving Average {0} Days, ppbv)'.format(w/24)
	ax.set_xlabel('Hours Since Jan 1st 2007')
	ax.set_ylabel(ylabel)
	ax.set_title('Fit')
	ax.plot(yw, label='testing data', alpha=0.5)
	ax.plot(zw, label='network estimate')
	plt.legend()
	f.savefig('fit_window_{0}.png'.format(str(w/24)), bbox_inches='tight')

plt.show()