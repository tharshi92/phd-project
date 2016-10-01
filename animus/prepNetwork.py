from config import *
import numpy as np
import random

print('Preparing Network Structure..')

time = np.arange(0, 24 * d).reshape((24 * d, 1))
state = time

for metadata in metadatum:

    title = metadata[1]
    if title != 'COField':
        data = np.load(title + '.npy')
        state = np.hstack((state, data))

    else:
        field = np.load(title + '.npy').reshape((len(state), 1))

mu_x = np.mean(state, axis=0)
s_x = np.std(state, axis=0, ddof=1)
mu_y = np.mean(field, axis=0)
s_y = np.std(field, axis=0, ddof=1)
scale_params = [mu_x, s_x, mu_y, s_y]

x = (state - mu_x)/s_x
y = (field - mu_y)/s_y

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
start = 365 * num + 366 * numLeap
end = start + 365 * (1 - leap) + 366 * leap

testingData = np.hstack((x[start:end, :], y[start:end, :]))
trainingData = np.hstack(np.ystack(x[:start, :], x[:end, :]), \
	np.ystack(y[:start, :], y[:end, :]))

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
np.save(homeDir + 'binaryData/scale_params.npy', scale_params)
print("Done.")
