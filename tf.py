from __future__ import print_function

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.05
training_epochs = 200
display_step = 1

# Network Parameters
n_hidden_1 = 3 # 1st layer number of features
n_input = 2
n_output = 1

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Store layers weight & bias
weights = {
    'h': tf.Variable(tf.random_normal([n_input, n_hidden_1], \
                                      stddev=1./np.sqrt(n_input))),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_output], \
                                        stddev=1./np.sqrt(n_hidden_1)))
}
biases = {
    'b': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Create model
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
cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
batch_x = np.array(([[0,0],[0,1],[1,0],[1,1]]))
batch_y = np.array([[1], [0], [0], [1]])
# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = 1
    # Loop over all batches
    for i in range(total_batch):
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                      y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
print("Optimization Finished!")

test = sess.run(pred, feed_dict={x:np.array([[0.9,-0.1]])})
print(test)
