from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Data
xTrain = np.load('x.npy')
yTrain = np.load('y.npy')
xTest = np.load('xt.npy')
yTest = np.load('yt.npy')

# Parameters
learning_rate = 1e-4
reg = 0.
training_epochs = 10000
display_step = 1000

# Network Parameters
n_hidden = 20
n_input = 8
n_output = 1

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Store layers weight & bias
weights = {
    'h': tf.Variable(tf.random_normal([n_input, n_hidden], \
                                      stddev=1./np.sqrt(n_input))),
    'out': tf.Variable(tf.random_normal([n_hidden, n_output], \
                                        stddev=1./np.sqrt(n_hidden)))
}
biases = {
    'b': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Create model and use a name scope to organize in visualizer
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h']), biases['b'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred - y)) + \
        reg * tf.nn.l2_loss(weights['h']) + \
        reg * tf.nn.l2_loss(weights['out'])
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#%%
# Initializing the variables
init = tf.initialize_all_variables()
    
# Launch the graph
sess = tf.Session()
sess.run(init)

J = []
J_test = []

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    avg_testcost = 0
    total_batch = 1
    # Loop over all batches
    for i in range(total_batch):
        # Run optimization op (backprop) and cost op (to get loss value)
        feed = {x: xTrain, y: yTrain}
        feed2 = {x: xTest, y: yTest}
        _, c = sess.run([optimizer, cost], feed_dict=feed)
        ct = sess.run(cost, feed_dict=feed2)
        # Compute and store average loss
        avg_cost += c / total_batch
        avg_testcost += ct / total_batch
        J.append(avg_cost)
        J_test.append(avg_testcost)
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
print("Optimization Finished!")
#%%

predTest = sess.run(pred, feed_dict={x:xTest})

# Plot Training History
cost_fig = plt.figure()
ax = plt.subplot(111)
plt.margins(0.0, 0.1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training History')
ax.loglog(J, label='Training')
ax.loglog(J_test, label='Testing')
plt.legend()

r = predTest - yTest

t = np.arange(0, len(yTest))/24

err = np.linalg.norm(r**2)/len(r)
m = np.float(np.mean(r, axis=0))
std = np.std(r, ddof=1)

f_CO = plt.figure()
ax = plt.subplot(111)
ylabel = 'CO Field (ppbv)'
ax.set_xlabel('Days Since Jan 1st 2007')
ax.set_ylabel(ylabel)
ax.set_title('Fit')
ax.plot(t, yTest, label='testing data')
ax.plot(t, predTest, label='network estimate')
ax.plot(t, r, label='residuals')
plt.legend(loc='center left')

f_hist = plt.figure()
ax = plt.subplot(111)  
title = 'MSE = {:.3f}, $\sigma$ = {:.3f}, $\mu$ = {:.3f}'
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title(title.format(err, std, m))
hist = plt.hist(r, alpha=0.5)


#%%













































