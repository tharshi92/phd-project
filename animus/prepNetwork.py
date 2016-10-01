from config import *
import numpy as np
import random
import pickle

print('Preparing Network Structure..')

time = np.arange(0, 24 * d).reshape((24 * d, 1))
state = time

for metadata in metadatum:

    title = metadata[1]
    if title != 'COField':
        data = np.load(homeDir + 'binaryData/' + title + '.npy')
        state = np.hstack((state, data))

    else:
        field = np.load(homeDir + 'binaryData/' + title + '.npy').reshape((len(state), 1))

mu_x = np.mean(state, axis=0)
s_x = np.std(state, axis=0, ddof=1)
yNorm = np.amax(field, axis=0) * 0.8
scale_params = [mu_x, s_x, yNorm]

x = (state - mu_x)/s_x
y = field/yNorm

testingYear = 2007
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

testingData = np.hstack((x[start:end, :], y[start:end, :]))
trainingData = np.hstack((np.delete(x, range(start, end) , axis=0), np.delete(y, range(start, end), axis=0)))

# shuffle x and y training arrays
seed = int(27)
new_order = list(range(len(trainingData)))
random.seed(seed)
random.shuffle(new_order)

trainingData = trainingData[new_order, :]

np.save(homeDir + 'binaryData/x.npy', x)
np.save(homeDir + 'binaryData/y.npy', y)
np.save(homeDir + 'binaryData/trData.npy', trainingData)
np.save(homeDir + 'binaryData/teData.npy', testingData) 
pickle.dump(scale_params, open(homeDir + 'binaryData/scaleParams.p', 'wb'))
print("Done.")
