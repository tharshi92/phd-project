import numpy as np
import random
import pickle
import sys

print('Preparing Network Structure..')

state = np.load('state.npy')[:365 * 2 * 24, :]
field = np.load('field.npy')[:365 * 2 * 24, :]

mu_x = np.mean(state[:8760, :], axis=0)
s_x = np.std(state[:8760, :], axis=0, ddof=1)
yNorm = np.amax(field[:8760, :], axis=0) * 0.8
scale_params = [mu_x, s_x, yNorm]

x = (state - mu_x)/s_x
y = field/yNorm

testingData = np.hstack((x[8760:, :], y[8760:, :]))
trainingData = np.hstack((x[:8760, :], y[:8760, :]))

# shuffle x and y training arrays
seed = int(27)
new_order = list(range(len(trainingData)))
random.seed(seed)
random.shuffle(new_order)

trainingData = trainingData[new_order, :]

np.save('trData.npy', trainingData)
np.save('teData.npy', testingData) 
pickle.dump(scale_params, open('scaleParams.p', 'wb'))
print("Done.")
