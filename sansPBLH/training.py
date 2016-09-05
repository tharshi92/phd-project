# Import 3rd Party Modules
import numpy as np
import matplotlib.pyplot as plt
from nami import Network, Trainer
from init_plotting import *
from config import nn, reg

# Load Training/Testing Data
x = np.load('x.npy')
x_test = np.load('xt.npy')
y = np.load('y.npy')
y_test = np.load('yt.npy')

# Network Parameters
N = len(x)
layers = [len(x.T), nn, len(y.T)]

# Create Network and Trainer Instances
net = Network(layers, N, reg, io=True)
trainer = Trainer(net)

#%%

# Train Network
trainer.train(x, y, x_test, y_test, method='BFGS')

# Save Final Weights
weights = net.get_params()
np.save('weights', weights)

#%%

# Plot Training History
init_plotting()
plt.cla()
ax = plt.subplot(111)
f_c = plt.gcf()
plt.margins(0.0, 0.1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training History')
ax.loglog(trainer.J, label='Training')
ax.loglog(trainer.J_test, label='Testing')
plt.legend()
#savename = 'costsnnco.pdf'
#plt.savefig(savename, bbox_inches='tight')
plt.show()
