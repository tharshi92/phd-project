# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 12:21:14 2016

@author: tharshi
"""
import matplotlib.pyplot as plt

def plotStreams(x):
    for i in range(len(x.T)):
        figPS = plt.figure()
        plt.plot(x[:, i])
        plt.title('data {}'.format(i))
        plt.xlabel('Sample')
        plt.ylabel('Value')
	plt.savefig('{0}.png'.format(i))
