
# coding: utf-8

# In[2]:

# Import modules
import numpy as np
import tensorflow as tf

import sys
import os.path
from time import time
from time import gmtime, strftime
import pickle
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from config import *
cwd = os.getcwd()

# Start an interactive session
sess = tf.InteractiveSession()


# In[4]:

# change to data directory
local_repo = home_dir + "18level" + '/'

os.chdir(local_repo)

# Reset tf graph
tf.reset_default_graph()

# Load Training/Testing Data
x = np.load('trData.npy')[:, :-1]
x_t = np.load('teData.npy')[:, :-1]
y = np.load('trData.npy')[:, -1].reshape((len(x), 1))
y_t = np.load('teData.npy')[:, -1].reshape((len(x_t), 1))
scale_params = pickle.load(open('scaleParams.p', 'rb'))

n_features = len(x.T)
n_targets = len(y.T)


# In[5]:

# Create placeholders for the input 
# and the target
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, n_features])
    y_ = tf.placeholder(tf.float32, shape=[None, n_targets])


# In[3]:

# Create network helper functions

def create_weights(input_dim, n_units, name):
    """A helper function for creating weights."""
    with tf.name_scope("Weights for Layer " + str(name)):
        return tf.Variable(tf.random_normal(shape=[input_dim, n_units],nstddev=1.0/input_dim))
    
def create_biases(n_biases, name):
    """A helper function for creating biases."""
    with tf.name_scope("Biases for Layer " + str(name)):
        return tf.Variable(tf.zeros(shape=[n_biases]))
    
def ff_layer(signal, n_units, num, act='relu'):
    """Send a signal through a layer of neuron
    units and returns the activations and weights if needed.
    
    """
    
    # Create weights and biases
    W = create_weights(signal.shape[0], n_units, num)
    b = create_biases(signal.shape[0], num)
    
    return tf.nn.relu(tf.matmul(signal, W) + b)


# In[13]:

# Initialize all tf variables
sess.run(tf.global_variables_initializer())
z1 = z.eval(feed_dict={v: x_t})
plt.figure(figsize=(18, 10))
plt.plot(y_t, label='Test')
plt.plot(z1, label='Net')
plt.legend()


# In[ ]:

# Train the model using gradient descent with a step size of 0.5
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

# Run training step multiple times to reduce loss
for i in range(len(x)):
    train_step.run(feed_dict={v: x[i, :].reshape((1, 8)), z_: y[i, :].reshape((1, 1))})


# In[ ]:

z2 = z.eval(feed_dict={v: x_t})
plt.figure(figsize=(18, 10))
plt.plot(y_t, label='Test')
plt.plot(z2, label='Net')
plt.legend()


# In[ ]:

e1 = loss.eval(feed_dict={v: x, z_: y})


# In[ ]:

e2 = loss.eval(feed_dict={v: x_t, z_: y_t})


# In[ ]:

print e1, e2

