
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

# Start an interactive session
sess = tf.InteractiveSession()


# In[2]:

# change to data directory
local_repo = home_dir + "18level" + '/'
os.chdir(local_repo)

# Load Training/Testing Data
x = np.load('trData.npy')[:, :-1]
x_t = np.load('teData.npy')[:, :-1]
y = np.load('trData.npy')[:, -1].reshape((len(x), 1))
y_t = np.load('teData.npy')[:, -1].reshape((len(x_t), 1))
scale_params = pickle.load(open('scaleParams.p', 'rb'))

n = len(x)
n_features = len(x.T)
n_targets = len(y.T)


# In[3]:

# Create placeholders for the input 
# and the target
with tf.name_scope('input'):
    x_ = tf.placeholder(tf.float32, shape=[None, n_features])
    y_ = tf.placeholder(tf.float32, shape=[None, n_targets])


# In[4]:

# Create network helper functions

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    #tf.summary.scalar('max', tf.reduce_max(var))
    #tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  """
  with tf.name_scope(layer_name):
        
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
        
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
        
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
        
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    
    return activations


# In[5]:

# Use the previous functions to create a model
fc1 = nn_layer(x_, n_features, 5, 'layer1')
fc2 = nn_layer(fc1, 5, 3, 'layer2')
z = nn_layer(fc2, 3, 1, 'output', act=tf.identity)

# Define the loss function
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.l2_loss(z - y_))
tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)


# In[6]:

# Merge all the summaries and write
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(save_dir + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(save_dir + '/test')

# Initialize all tf variables
sess.run(tf.global_variables_initializer())

# Writer for logs
writer = tf.summary.FileWriter(save_dir, graph=tf.get_default_graph())

for i in range(200):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, err = sess.run([merged, loss], feed_dict={x_: x_t, y_: y_t})
    test_writer.add_summary(summary, i)
    print('Error at step %s: %s' % (i, err))
  else:  # Record train set summaries, and train
    summary, _ = sess.run([merged, train_step], feed_dict={x_: x, y_: y})
    train_writer.add_summary(summary, i)


# In[7]:

z1 = z.eval(feed_dict={x_: x_t})
plt.figure(figsize=(20, 10))
plt.plot(y_t, label='Test')
plt.plot(z1, label='Net')
plt.legend()
plt.savefig(save_dir + '/fit1.png')


# In[ ]:



