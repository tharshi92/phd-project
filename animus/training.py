# Import 3rd Party Modules
import numpy as np
import matplotlib.pyplot as plt
from nami import Network, Trainer
from config import *
import sys

localRepo = homeDir + sys.argv[1] + '/'

# change to data directory
os.chdir(localRepo)

# Load Training/Testing Data
x = np.load('trData.npy')[:, :-1]
x_test = np.load('teData.npy')[:, :-1]
y = np.load('trData.npy')[:, -1].reshape((len(x), 1))
y_test = np.load('teData.npy')[:, -1].reshape((len(x_test), 1))

# Network Parameters
N = len(x)
layers = [len(x.T), hiddenNeurons, len(y.T)]

# Create Network and Trainer Instances
net = Network(layers, N, reg, io=True)
trainer = Trainer(net)

#%%

# Train Network
trainer.train(x, y, x_test, y_test, method='BFGS')

# change to save directory
os.chdir(saveDir)

# secondary directory for different runs on sameday
if not os.path.exists(sys.argv[1]):
    os.makedirs(sys.argv[1])

# change to specific run directory
os.chdir(sys.argv[1])

# Save Final Weights
weights = net.get_params()
np.save('weights', weights)

#%%

# Plot Training History
f = plt.figure()
ax = plt.subplot(111)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training History')
ax.loglog(trainer.J, label='Training')
ax.loglog(trainer.J_test, label='Testing')
plt.legend()
savename = 'costsnnco.pdf'
plt.savefig(savename, bbox_inches='tight')
plt.show()
