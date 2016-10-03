# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from nami import Network
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
os.chdir(saveDir + sys.argv[1])

# Create Network and Trainer Instances
net = Network(layers, N, reg)
wbs = np.load('weights.npy')
net.set_params(wbs)

yp = sP[2] *  y
z = sP[2] * net.forward(x)
r = z - yp

t = np.arange(0, len(yp))/24

err = np.linalg.norm(r**2)/len(r)
m = np.float(np.mean(r, axis=0))
std = np.std(r, ddof=1)

f0 = plt.figure()
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'
ax.set_xlabel('Days Since Jan 1st 2006')
ax.set_ylabel(ylabel)
ax.set_title('Fit')
ax.plot(t, yp, label='testing data')
ax.plot(t, z, label='network estimate')
ax.plot(t, r, label='residuals')
plt.legend(loc='center left')
f0.savefig('fit.pdf', bbox_inches='tight')

f = plt.figure()
ax = plt.subplot(111)  
title = 'MSE = {:3f}, $\sigma$ = {:3f}, $\mu$ = {:3f}'
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title(title.format(err, std, m))
hist = plt.hist(r, alpha=0.5)
f.savefig('hist.pdf', bbox_inches='tight')
plt.show()
