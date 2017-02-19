
# coding: utf-8

# In[1]:

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


# In[2]:

local_repo = home_dir + "18level" + '/'

# change to data directory
os.chdir(local_repo)

# Load Training/Testing Data
x = np.load('trData.npy')[:, :-1]
x_t = np.load('teData.npy')[:, :-1]
y = np.load('trData.npy')[:, -1].reshape((len(x), 1))
y_t = np.load('teData.npy')[:, -1].reshape((len(x_t), 1))
scale_params = pickle.load(open('scaleParams.p', 'rb'))

n_features = len(x.T)
n_targets = len(y.T)


# In[15]:

# Create placeholders for the input 
# and the target
v = tf.placeholder(tf.float32, shape=[None, n_features])
z_ = tf.placeholder(tf.float32, shape=[None, n_targets])

# Variables in our model (weights and biases)
W1 = tf.Variable(tf.random_normal([n_features, 10]))
b1 = tf.Variable(tf.random_normal([10]))

W2 = tf.Variable(tf.random_normal([10, n_targets]))
b2 = tf.Variable(tf.random_normal([n_targets]))

# Initialize all tf variables
sess.run(tf.global_variables_initializer())

# This is our regression model
tmp = tf.nn.relu(tf.matmul(v, W1) + b1)
z = tf.matmul(tmp, W2) + b2

# Define a loss function
loss = tf.reduce_mean(tf.nn.l2_loss(z - z_))


# In[23]:

z1 = z.eval(feed_dict={v: x_t})
plt.plot(scale_params[2] * y_t)


# In[ ]:

# Train the model using gradient descent with a step size of 0.5
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

# Run training step multiple times to reduce loss
for i in range(int(2e3)):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_:batch[1]})


# In[ ]:



