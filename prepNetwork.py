"""
Created on Fri Jul 01 17:05:54 2016

@author: tharshi sri, tsrikann@physics.utoronto.ca
"""

import numpy as np
import matplotlib.pyplot as plt
import random

print('-------------------------------------------------')
print('Preparing Network Structure..')
print('-------------------------------------------------\n')

time = np.arange(0, 24*365).reshape((24*365, 1))

data = np.hstack((\
    time,\
    np.load('trUWind.npy'),\
    np.load('trVWind.npy'),\
    np.load('trPressure.npy'),\
    np.load('trTemperature.npy'),\
    np.load('trHumidity.npy'),\
    np.load('trPBL.npy'),\
    np.load('trCOSource.npy')))
    
data_test = np.hstack((\
    time,\
    np.load('teUWind.npy'),\
    np.load('teVWind.npy'),\
    np.load('tePressure.npy'),\
    np.load('teTemperature.npy'),\
    np.load('teHumidity.npy'),\
    np.load('tePBL.npy'),\
    np.load('teCOSource.npy')))
    
target = np.load('trCOField.npy').reshape((len(data), 1))
target_test = np.load('teCOField.npy').reshape((len(data_test), 1))

mu_x = np.mean(data, axis=0)
s_x = np.std(data, axis=0, ddof=1)
scale_params = [mu_x, s_x]

# shuffle x and y arrays

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
