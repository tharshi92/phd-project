# Import 3rd Party Modules
import numpy as np
import matplotlib.pyplot as plt
from nami import Network, Trainer
from config import *

# change to data directory
os.chdir(homeDir + 'binaryData/')

# Load Training/Testing Data
x = np.load('trData.npy')[:, :-1]
x_test = np.load('testingData.npy')[:, :-1]
y = np.load('trData.npy')[:, -1]
y_test = np.load('testingData.npy')[:, :-1]

# Network Parameters
N = len(x)
layers = [len(x.T), 4, len(y.T)]

# Create Network and Trainer Instances
reg = 1e-4
net = Network(layers, N, reg, io=True)
trainer = Trainer(net)

#%%

# Train Network
trainer.train(x, y, x_test, y_test, method='L-BFGS-B')

# change to save directory
os.chdir(saveDir)
# Save Final Weights
weights = net.get_params()
np.save('weights', weights)

#%%

# Plot Training History
f = plt.figure()
ax = plt.subplot(111)
plt.margins(0.0, 0.1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training History')
ax.loglog(trainer.J, label='Training')
ax.loglog(trainer.J_test, label='Testing')
plt.legend()
savename = 'costsnnco.png'
plt.savefig(savename, bbox_inches='tight')
plt.show()
