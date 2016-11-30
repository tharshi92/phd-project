# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:48:54 2016

@author: tharshi
"""
# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from config import *    
import sys

localRepo = homeDir + sys.argv[1] + '/'

# change to data directory
os.chdir(localRepo)

y = np.load('y.npy')
t = np.load('t.npy')

def freqSpec(s, timeStep):
    ps = np.abs(np.fft.fft(s))**2
    freqs = np.fft.fftfreq(s.size, timeStep)
    idx = np.argsort(freqs)
    return idx, freqs, ps

s = y
timeStep = t[1] - t[0]

idx, freqs, ps = freqSpec(s, timeStep)

plt.subplot(211)
plt.plot(t, y)
plt.xlabel('Days Since 010106')
plt.ylabel('CO Field')
plt.subplot(212)
plt.plot(freqs[idx], ps[idx])
plt.ylabel('ps')
plt.xlabel('Frequency 1/Days')
plt.savefig('fft.png')
print('Max Amplitude Freq: {0}'.format(freqs[np.argmax(ps)]))

plt.show()