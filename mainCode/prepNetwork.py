from config import *
import numpy as np
import random
import pickle
import sys

localRepo = homeDir + sys.argv[1] + '/'

print('Preparing Network Structure..')

time = np.arange(0, 24 * d).reshape((24 * d, 1)) % 24
state = time

for metadata in metadatum:

    title = metadata[1]
    if title != 'COField':
        data = np.load(localRepo + title + '.npy')
        state = np.hstack((state, data))

    else:
        field = np.load(localRepo + title + '.npy').reshape((len(state), 1))

state = np.hstack((state, np.load(localRepo + 'COSource' + '.npy')))

leap = 0
flag = (testingYear - 2006 + 2) % 4
if flag == 0:
	leap = 1
numLeap = 0
for i in range(testingYear - 2006):
	if (i + 2) % 4:
		numLeap += 1
num = (testingYear - 2006) - numLeap
start = 24 * (365 * num + 366 * numLeap)
end = start + 24 * (365 * (1 - leap) + 366 * leap)

mu_x = np.mean(np.delete(state, range(start, end) , axis=0), axis=0)
s_x = np.std(np.delete(state, range(start, end) , axis=0), axis=0, ddof=1)
yNorm = 1.0 * np.amax(np.delete(field, range(start, end) , axis=0), axis=0)
scale_params = [mu_x, s_x, yNorm]

x = (state - mu_x)/s_x
y = field/yNorm
t = np.linspace(0, len(y)/24, len(y))

testingData = np.hstack((x[start:end, :], y[start:end, :]))
trainingData = np.hstack((np.delete(x, range(start, end) , axis=0), np.delete(y, range(start, end), axis=0)))

# shuffle x and y training arrays
seed = int(27)
new_order = list(range(len(trainingData)))
random.seed(seed)
random.shuffle(new_order)

trainingData = trainingData[new_order, :]

np.save(localRepo + 't.npy', t)
np.save(localRepo + 'x.npy', x)
np.save(localRepo + 'y.npy', y)
np.save(localRepo + 'trData.npy', trainingData)
np.save(localRepo + 'teData.npy', testingData) 
pickle.dump(scale_params, open(localRepo + 'scaleParams.p', 'wb'))
print("Done.")
