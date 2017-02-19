import numpy as np
import matplotlib.pyplot as plt

def plotStreams(x):
    for i in range(len(x.T)):
        figPS = plt.figure()
        plt.plot(x[:, i])
        plt.title('data {}'.format(i))
        plt.xlabel('Sample')
        plt.ylabel('Value')
	plt.savefig('{0}.png'.format(i))

def moving_average(a, window) :
    ma = np.cumsum(a, dtype=float)
    ma[window:] = ma[window:] - ma[:-window]
    return ma[window - 1:] / window
