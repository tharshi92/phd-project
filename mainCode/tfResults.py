from __future__ import print_function
from config import *
import sys
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# change to specific run directory
os.chdir(saveDir + sys.argv[1] + '/tf')

z = np.load('z.npy')
y = np.load('signal.npy')
r = z - y

t = np.arange(0, len(y))/24
err = np.linalg.norm(r**2)/len(r)
m = np.float(np.mean(r, axis=0))
std = np.std(r, ddof=1)