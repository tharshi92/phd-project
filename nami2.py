import numpy as np

class Network(object):
    
    def __init__(self, xTrain, numNodes, yTrain):
        self.N = len(xTrain)
        self.x = xTrain
        self.y = yTrain
        self.data = np.hstack((x, y))
        self.numNodes = numNodes
        self.w1 = np.random.normal(scale = 1.0/np.sqrt(len(self.x.T)), \
            size = (len(self.x.T), self.numNodes))
        self.b1 = np.random.randn(1, self.numNodes)
        self.b2 = np.random.randn(1, len(self.y.T))
        self.w2 = np.random.normal(scale = 1.0/np.sqrt(len(self.w1)), \
            size = (self.numNodes, len(self.y.T)))
        
    def nl(self, z):
        return 1.7159*np.tanh(2*z/3)
    
    def nlPrime(self, z):
        return 1.14393*(1.0 - np.tanh(2*z/3)**2)
    
    def forward(self, x, retall = False):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.nl(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.nl(z2)
        if retall:
            return z1, a1, z2, a2
        return a2
    
    def cost(self, x, y):
        a2 = self.forward(x)
        d = a2 - y
        c = np.float(np.sum(d**2, axis=0)/(2.0 * self.N))
        return c
        
    def backprop(self, x, y):
        z1, a1, z2, a2 = self.forward(x, retall = True)
        cnst = 1.0/self.N
        diff = a2 - y
        delta2 = cnst * diff * self.nlPrime(z2)
        dw2 = np.dot(a1.T, delta2)
        delta1 = np.dot(delta2, self.w2.T) * self.nlPrime(z1)
        dw1 = np.dot(x.T, delta1)
        db1 = np.sum(delta1, axis=0)
        db2 = np.sum(delta2, axis=0)
        return dw1, db1, dw2, db2
        
    def train(self, numIter, numBatches, rate, xTest, yTest):
        self.J = []
        self.JTest = []
        self.xt = xTest
        self.yt = yTest
        batches = np.split(self.data, numBatches, axis=0)
        count = 1
        try:
            for batch in batches:
                print('Batch #{}'.format(count))
                count += 1
                X = batch[:, len(x.T):]
                Y = batch[:, len(x.T):]
                for i in range(numIter):
                    #change weights
                    dw1, db1, dw2, db2 = self.backprop(X, Y)
                    delta_dw1_new = rate * dw1
                    delta_db1_new = rate * db1
                    delta_dw2_new = rate * dw2
                    delta_db2_new = rate * db2
                    self.w1 -= delta_dw1_new
                    self.w2 -= delta_dw2_new
                    self.b1 -= delta_db1_new
                    self.b2 -= delta_db2_new
                    #update cost
                    c = self.cost(self.x, self.y)
                    cTest = self.cost(self.xt, self.yt)
                    self.J.append(c)
                    self.JTest.append(cTest)
                    mod = numIter/100
                    if i % mod == 0:
                        print(c, cTest)
        except KeyboardInterrupt:
            pass
        return self.J, self.JTest
#%%       
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    M = 10
    x = np.random.uniform(-10, 10, M).reshape(M, 1)
    xt = np.random.uniform(-10, 10, M).reshape(M, 1)
    y = (2 * x + 1).reshape((len(x), 1))
    yt = (2 * xt + 1).reshape((len(xt), 1))
    mu = np.mean(x, axis=0)
    s = np.std(x, axis=0, ddof=1)
    m = np.amax(y)
    
    #%%
    n = Network((x - mu)/s, 10, y/m)
    J, Jt = n.train(int(1e5), M, 1e-6, (xt - mu)/s, yt/m)
    #%%
    plt.figure()
    plt.loglog(J)
    plt.loglog(Jt)
    
    t = np.linspace(-20, 20, 1000).reshape((1000, 1))
    truth = (2*t + 1).reshape((len(t), 1))
    fit = m * n.forward((t - mu)/s)
    plt.figure()
    plt.plot(t, fit)
    plt.plot(t, truth)
    plt.plot(xt, yt, 'k*')