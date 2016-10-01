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

mu_x = np.mean(data, axis=0)
s_x = np.std(data, axis=0, ddof=1)
mu_y = np.mean(target, axis=0)
s_y = np.std(target, axis=0)
scale_params = [mu_x, s_x, mu_y]

# shuffle x and y training arrays
seed = int(27)
new_order = list(range(len(data)))
random.seed(seed)
random.shuffle(new_order)

x = (data[new_order, :] - mu_x)/s_x
y = target[new_order, :]
x_test = (data_test - mu_x)/s_x
y_test = target_test

np.save('x', x)
np.save('y', y)   
np.save('xt', x_test)
np.save('yt', y_test)
np.save('data', data)
np.save('data_test', data_test)
np.save('target', target)
np.save('target_test', target_test)
np.save('scale_params', scale_params)
print("Done.")
