# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 12:12:50 2016

@author: tharshi sri, tsrikann@physics.utoronto.ca

nami.py: A feed forward neural network with backpropagation and multiple methods
of minimization. This network was designed for use in regression problems. The
program is not optimized for GPU usage. Classification is currently in progress.

"""

# Import 3rd party packages
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt

class Network(object):
    
    def __init__(self, layers, N, reg=0, io=False):
        """Initializes the topology of the network and sets up random weights
        for each layer. Note that h_layers does not have to include the input
        and output layer.
        
        layers : A list of integers where each int represents the number of 
        non-bias neurons in the layer.
        
        x : The input data matrix. x should be dimension(x) by (number of
        examples)
        
        y: The input data matrix. y should be dimension(y) by (number of
        examples)
        
        reg: The amount of L2 regularization to use on the cost function.
        
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.N = N
        self.reg = reg
        # initialize weights and biases
        self.sizes = list(zip(self.layers[:-1], self.layers[1:]))
        self.weights = [np.random.normal(scale = np.sqrt(1.0/l), \
                size=(s[0], s[1])) \
                for s, l in list(zip(self.sizes, self.layers[:-1]))]
        self.biases = [np.random.randn(1, t) for t in layers[1:]]
        self.weight_shapes = [w.shape for w in self.weights]
        self.bias_shapes = [b.shape for b in self.biases]
        self.num_weights = sum([int(r*s) for r, s in self.weight_shapes])
        self.num_biases = sum([int(t*u) for t, u in self.bias_shapes])
        self.num_params = self.num_weights + self.num_biases
        # initialize derivative matrices
        self.dws = [np.zeros((r, s)) for r, s in self.sizes]
        self.dbs = [np.zeros((1, t)) for t in layers[1:]]
        
        if io:
            print('################################################')
            print('Feed Forward Neural Network for Regression')
            print('################################################')
            print('number of layers: {0}'.format(self.num_layers))
            print('number of training examples: {0}'.format(self.N))
            print('regularization parameter: {0}'.format(self.reg))
            print('weight shapes: {0}'.format(self.weight_shapes))
            print('bias shapes: {0}'.format(self.bias_shapes))
            print('number of parameters (Ws + Bs): {0} + {1} = {2}'\
                .format(self.num_weights, self.num_biases, self.num_params))
            print('------------------------------------------------\n')
        
    def plot_weights(self, plot_type='image', init_label=0):
        w_idx = 2
        prefix = '_'
        if init_label:
            prefix = 'init_'
        if plot_type == 'image':
            for w in self.weights:
                fw = plt.figure()
                im = plt.imshow(w, interpolation='none', cmap='viridis')
                cbar = plt.colorbar(im, pad=0.05)
                title = 'W{0}_{1}'.format(w_idx, self.layers, self.reg)
                plt.title(title)
                savename  = prefix + title + '_weight_images'
                plt.savefig(savename, extension='png', dpi=300)
                w_idx += 1
        if plot_type == 'histogram':
            for w in self.weights:
                fw = plt.figure()
                im = plt.hist(w.flatten(), alpha=0.5, bins=100)
                title = prefix + 'W{0}_{1}'.format(w_idx, self.layers, self.reg)
                plt.title(title)
                savename  = title + '_weight_hists'
                plt.savefig(savename, extension='png', dpi=300)
                w_idx += 1
    
    def g(self, z):
        """Return nonlinear transformation to neuron output."""
        return 1.7159*np.tanh(2*z/3)
        
    def g_prime(self, z):
        """Return derivative of nonlinear activation function."""
        return 1.14393*(1 - np.tanh(2*z/3)**2)
        
    def h(self, z):
        """Return the output transformation."""
        return z
            
    def h_prime(self, z):
        """Return the derivative of the output transformation."""
        return 1

    def forward(self, x, neuron_info=0):
        """Return the final activation of the input matrix x. If neuron_info is
        on then a list on neuron inputs and outputs are returned in a list per
        layer.
        
        """
        zs = []
        acts = [x]
        a = x
        for w, b in list(zip(self.weights[:-1], self.biases[:-1])):
            # notice that + b underneath is broadcast row-wise
            z = np.dot(a, w) + b
            zs.append(z)
            a = self.g(z)
            acts.append(a)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self.h(z)
        acts.append(a)
        if neuron_info:
            ans = a, zs, acts[:-1]
        else:
            ans = a
        return ans
        
    def cost(self, x, y):
        """Return the squared cost with optional L2 regularization for 
        data {x, y}.
        
        """
        yhat = self.forward(x)
        diff = yhat - y
        ps = self.get_params()
        J_0 = np.sum(diff**2, axis=0)
        J_r = self.reg*np.sum(ps[:self.num_weights]**2)
        return 0.5*(J_0 + J_r)/self.N
    
    def cost_prime(self, x, y):
        """return the lists of derivatives dNet/dW and dNet/dB."""
        yhat, zs, acts = self.forward(x, neuron_info=1)
        cnst = 1.0/self.N
        diff = yhat - y
        delta = cnst*diff*self.h_prime(zs[-1])
        self.dws[-1] = (np.dot(acts[-1].T, delta) \
                + (self.reg/self.N)*self.weights[-1])
        self.dbs[-1] = np.sum(delta, axis=0)
        w_previous = self.weights[-1]
        # loop backwards and calculate the rest of the derivatives !!!
        for i in range(self.num_layers - 2, 0, -1):
            w_current = self.weights[i - 1]
            w_previous = self.weights[i]
            z = zs[i - 1]
            a = acts[i - 1]
            delta = np.dot(delta, w_previous.T)*self.g_prime(z)
            self.dws[i - 1] = np.dot(a.T, delta) \
                + (self.reg/self.N)*w_current
            self.dbs[i - 1] = np.sum(delta, axis=0)
            w_previous = w_current
        return self.dws, self.dbs
        
    def set_params(self, params, retall=0):
        """Change single parameter array into weight/bias matricies."""
        idx = 0
        start = 0
        for w in self.weights:
            temp = self.sizes[idx][0]
            tempp = self.sizes[idx][1]
            end = start + temp*tempp
            self.weights[idx] = params[start:end].reshape((temp, tempp))
            start = end
            idx += 1
        idx = 0
        for b in self.biases:
            temppp = self.layers[1:][idx]
            end = start + temppp
            self.biases[idx] = params[start:end].reshape((1, temppp))
            start = end
            idx += 1
        if retall:
            return self.weights, self.biases
   
    def get_params(self):
        """Get all weights/biases rolled into one array."""
        params = self.weights[0].ravel()
        for w in self.weights[1:]:
            params = np.concatenate((params, w.ravel()))
        for b in self.biases:
            params = np.concatenate((params, b.ravel()))
        return params
     
    def get_grads(self, x, y):
        """Get all weights/biases rolled into one array."""
        dws, dbs = self.cost_prime(x, y)
        grads_list = [dw.ravel() for dw in dws] + [db.ravel() for db in dbs]
        grads = grads_list[0]
        for g in grads_list[1:]:
            grads = np.concatenate((grads, g))
        return grads
        
    def numgrad(self, x, y):
        """Get a numerical estimate of the gradient vector at (x,y)."""
        paramsInitial = self.get_params()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-5
    
        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            self.set_params(paramsInitial + perturb)
            loss2 = self.cost(x, y)
            
            self.set_params(paramsInitial - perturb)
            loss1 = self.cost(x, y)
    
            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)
    
            # Return the value we changed to zero:
            perturb[p] = 0
        
        # Return Params to original value:
        self.set_params(paramsInitial)
    
        return numgrad 

class Trainer(object):
    
    def __init__(self, Network):
        """Make Local reference to network."""
        self.net = Network
        
    def cost_wrapper(self, params, x, y):
        self.net.set_params(params)
        cost = self.net.cost(x, y)
        return cost
        
    def grad_wrapper(self, params, x, y):
        self.net.set_params(params)
        grads = self.net.get_grads(x, y)
        return grads
        
    def callback(self, params):
        self.net.set_params(params)
        self.J.append(self.net.cost(self.x, self.y))
        self.J_test.append(self.net.cost(self.x_t, self.y_t))   
        
    def train(self, x_train, y_train, x_test, y_test, method='L-BFGS-B'):
        # Make an internal variable for the callback function:
        self.x = x_train
        self.y = y_train
        self.x_t = x_test
        self.y_t = y_test
        self.method = method

        # Make empty list to store training/testing costs:
        self.J = []
        self.J_test = []
        
        print('Minimization using ' + self.method + ':')        
        
        params0 = self.net.get_params()
        options = {'disp': True, 'gtol': 1e-07, 'return_all': True}
        _res = spo.minimize(\
        	self.cost_wrapper, \
        	params0, method=self.method, \
        	jac=self.grad_wrapper, \
        	args=(x_train, y_train), 
        	callback=self.callback, \
        	options=options)
            
        self.net.set_params(_res.x)
        self.results = _res
        
        #%%
if __name__ == '__main__':
     
    N = 100
    reg = 1e-3
    
    t = np.random.uniform(low=230, high=310, size=(int(N), 1))
    t2 = np.random.uniform(low=230, high=310, size=(int(N), 1))
    d = np.random.uniform(low=0, high=365, size=(int(N), 1))
    d2 = np.random.uniform(low=0, high=365, size=(int(N), 1))
    
    temp1 = np.linspace(0, 365, 1000)
    temp2 = np.linspace(200, 350, 1000)
    dd, tt = np.meshgrid(temp1, temp2)
    
    def mice(d, t):
        return (50 * np.sin(-2*np.pi*(d + 180)/30) + 100) * \
            np.exp(-(t - 280)**2 /100000.0)
        
    truth = mice(dd, tt)
    y = np.zeros(int(N)).reshape((N, 1)) + 5*np.random.randn()
    y2 = np.zeros(int(N)).reshape((N, 1)) + 5*np.random.randn()
    
    for i in range(N):
        y[i] = mice(d[i], t[i])
        y2[i] = mice(d2[i], t2[i])
    
    x = np.hstack((d, t))
    x2 = np.hstack((d2, t2))
    
    mu_x = np.mean(x, axis=0)
    s_x = np.std(x, axis=0, ddof=1)
    mu_y = np.mean(y, axis=0)
    s_y = np.std(y, axis=0, ddof=1)

    layers = [len(x.T), 10, len(y.T)]

    net = Network(layers, N, reg, io=1)
    #%%
    trainer = Trainer(net)
    trainer.train((x - mu_x)/s_x, y, \
        (x2 - mu_x)/s_x, y2, method='L-BFGS-B')
        
    f_cost = plt.figure()
    ax = plt.subplot(111)
    ax.set_title('Training History')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.loglog(trainer.J, label='Training')
    plt.loglog(trainer.J_test, label='Testing')
    plt.legend()
    plt.show()



