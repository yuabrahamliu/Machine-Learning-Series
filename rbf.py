# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:58:03 2018

@author: Yu Liu
"""

#%%
import numpy as np
from numpy.linalg import norm
from random import random
from math import exp

def rand(a, b):
    """
    random value generated for parameter initialization
    Theta form of Lagrange's mean value theorem
    
    @param: a, b, the upper and lower limits of the random value
    @return: xi, random value
    """
    theta = random()
    xi = (b-a)*theta + a
    return xi

def RBF(x, beta, c):
    """
    rho(x, ci) = exp(-betai*||x-ci||**2)
    
    @param: x, array, input variable
    @param: beta, float, scale index
    @param: c, array, center
    @return: rho, RBF function output value
    """
    matnorm = norm(x-c, 2)
    #norm(x, ord=None, axis=None) return matrix or vector norm
    #For maxtrix, if ord = 2, return 2-norm 
    #(the sqrt of the maxium eign value of x.T.dot(x))
    #(the maxium sing. value)
    #For vector, if ord = 2, return 2-norm
    #(the length of the vector)
    index = -1*beta*matnorm*matnorm
    rho = exp(index)
    return rho

class RBP_network:
    
    def __init__(self):
        """
        Construction method of BP_network class
        """
        #neuron numbers of the input, hidden and output layers
        self.i_n = 0
        self.h_n = 0
        self.o_n = 0
        
        #Output value of the layers
        self.i_v = []
        self.b = [] #hidden layer
        self.y = 0.0
        
        #parameters (w, b, c)
        self.w = []
        #weights of the links between hidden layer neurons and output neuron
        #In RBF network, all the connection weights between input layer and 
        #hidden layer 1!
        #Because each hidden neuron will accept the whole input matrx as the 
        #input of its RBF function rho(x, ci). Note x has no index, meaning it 
        #is the whole input matrix.
        self.beta = []
        #scale indexes of Gaussian-RBFs of different hidden neurons
        #Each hidden neuron has its own beta
        self.c = []
        #centers of Gaussian-RBFs of different hidden neurons
        #Each hidden neuron has its own center
        
        #Initialize the learning rate
        self.lr = 0.0
        
    def CreateNN(self, ni, nh, centers, learningrate):
        """
        Build a RBP network structure and initialize parameters
        
        @param; ni, nh, the neuro numbers of the input and hidden layers
        @param: centers, maxtrix [h_n * i_n], the center parameters object to
        hidden layer neurons. For each hidden neuron, its center parameter is 
        one of the h_n rows of the center matrix. Because each sample input 
        has the dimension of [1 * i_n], where i_n is the number of input
        neurons, also the number of input attributes after one-hot coding, in 
        this case, for a hidden neruron, its input (the whole input matrix 
        with [1 * i_n] dimension), and its specific center paramter (also has 
        a dimension of [1 * i_n]), have the same shape)
        @param: learningrate, learning rate of gradient algorithm
        """
        #Assign the neuron numbers of the input, hidden and output layers
        self.i_n = ni
        self.h_n = nh
        self.o_n = 1 #Output layer contains only 1 neuron
        
        #Initialize output value of each layer
        self.i_v = np.zeros(self.i_n)
        self.b = np.zeros(self.h_n)
        
        #Initialize centers of Gaussian-RBFs of different hidden neurons
        self.c = centers
        
        #Initialize weight for each hidden-output connection 
        #(random initialization)
        #Initialize scale index for each hidden neuron
        self.w = np.zeros(self.h_n)
        self.beta = np.zeros(self.h_n)
        for h in range(self.h_n):
            self.w[h] = rand(0, 1)
            self.beta[h] = rand(0, 1)
        
        #Initialize learning rate
        self.lr = learningrate
        
    def Pred(self, x):
        """
        y = sigma(wi*rho(x, ci))
        
        @param: x, the input array for input layer (one sample, array). Sample 
        attributes served as input array, each attribute value VECTOR element 
        after dummy matrix transformation corresponds to an array element
        @param: y, float, output of the RBP network
        """
        self.y = 0.0
        #Must reset y to 0.0 first in this Pred method, because this method is 
        #used to perform on only one sample, if y is not set to 0.0 each time 
        #this function acts, the current sample will use the y value (output)
        #value of the former sample as a basis to add its output value and 
        #the final result will be the accumulative sum of all the output of 
        #all the samples.
        
        #Activate hidden layer and calculating output
        for h in range(self.h_n):
            self.b[h] = RBF(x, self.beta[h], self.c[h])
            self.y += self.w[h]*self.b[h]
            
        return self.y
    
    def Batch_Pred(self, x):
        """
        Predict process through the RBF network for batch data
        
        @param: x, Sample attribute data set (array). Each row 
        corresponds to a sample. Each attribute value VECTOR element after 
        dummy matrix transformation corresponds to a column
        @return y: array, output of the network, each element is the output 
        for one sample in the sample batch
        """
        y_pred = []
        #Activate hidden layer and calculate output
        for i in range(len(x)):
            y_pred.append(self.Pred(x[i]))
            
        return y_pred
    
    def BackPropagateRBF(self, x, y, method = 'standard'):
        """
        gh = (y(k)hat - y(k))*bh = (y(k)hat - y(k))*rho(x, ch)
        
        delta_betah = eta*gh*wh*||x-ch||**2
        
        delta_wh = -eta*gh
        
        @param: x, the input array for input layer (one sample, array). Sample 
        attributes served as input array, each attribute value VECTOR element 
        after dummy matrix transformation corresponds to an array element
        @param: y, sample label served as output(one sample, float)
        @param: method, string, type of BP algorithsm to be used, 'standard' or 
        'accumulative'
        """
        #Get current network output
        self.Pred(x)
        
        #Calculate the gradient for hidden layer
        g = np.zeros(self.h_n)
        for h in range(self.h_n):
            g[h] = (self.y - y)*self.b[h]
            
        #Update the parameter
        if method == 'accumulative':
            self.delta_beta = np.zeros(self.h_n)
            self.delta_w = np.zeros(self.h_n)
            
        for h in range(self.h_n):
            if method == 'standard':
                self.beta[h] += self.lr*g[h]*self.w[h]*(norm(x-self.c[h], 2))**2
                self.w[h] -= self.lr*g[h]
            else:
                self.delta_beta[h] += \
                self.lr*g[h]*self.w[h]*(norm(x-self.c[h], 2))**2
                self.delta_w[h] -= self.lr*g[h]                
            
    def TrainRBF(self, data_in, data_out, method = 'standard'):
        """
        Perform BP training for RBF network
        
        @param: data_in, Sample attribute data set (array). Each row 
        corresponds to a sample. Each attribute value VECTOR element after 
        dummy matrix transformation corresponds to a column
        @param: data_out, Sample label data set (array). Each element is the 
        label (float) for one sample in the sample batch.
        @param: method, string, type of BP algorithsm to be used, 'standard' or 
        'accumulative'
        @return: e, accumulated error
        @return: e_k, error list consisting of the error of all the samples
        """
        e_k = []
        samplesize = len(data_in)
        for k in range(samplesize):
            x = data_in[k]
            y = data_out[k]
            self.BackPropagateRBF(x, y, method = method)
            
            #Error in train set for each step
            y_delta2 = (self.y - y)**2
            e_k.append(y_delta2/2.0)
            
        #Total error of training
        e = sum(e_k)/len(e_k)
        
        if method == 'accumulative':
            self.beta += self.delta_beta/samplesize
            self.w += self.delta_w/samplesize
        
        return e, e_k
#%%
if __name__ == '__main__':
    #train set 
    X_trn = np.random.randint(0, 2, (100, 2))
    y_trn = np.logical_xor(X_trn[:, 0], X_trn[:, 1])
    #test set
    X_tst = np.random.randint(0, 2, (100, 2))
    y_tst = np.logical_xor(X_tst[:, 0], X_tst[:, 1])
    
    #Generate the centers (4 centers with 2 dimensions) based on XOR data
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#%%Train a RBF network using standard BP method    
    #Construct the RBF network
    rbf_nn = RBP_network()
    rbf_nn.CreateNN(ni=len(X_trn), nh=4, centers=centers, learningrate=0.05)
    
    #Parameter training using standard BP method
    e1 = []
    for i in range(10):
        err, err_k = rbf_nn.TrainRBF(data_in=X_trn, data_out=y_trn, 
                                     method='standard')
        e1.append(err)
        
    #model testing
    y_pred = rbf_nn.Batch_Pred(X_tst)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] >= 0.5:
            y_pred[i] = True
        else:
            y_pred[i] = False
        if y_pred[i] == y_tst[i]:
            count += 1
    tst_err = 1 - count/float(len(y_tst))
    print "RBF network trained by standard BP method test error rate: %.3f" % \
    tst_err
    
    del err, err_k, count, i
#%%Train a RBF network using accumulative BP method
    #Construct the RBF network
    rbf_nn = RBP_network()
    rbf_nn.CreateNN(ni=len(X_trn), nh=4, centers=centers, learningrate=0.05)
    
    #Parameter training using accumulative BP method
    e2 = []
    for i in range(10):
        err, err_k = rbf_nn.TrainRBF(data_in=X_trn, data_out=y_trn, 
                                      method='accumulative')
        e2.append(err)
        
    #model testing
    y_pred = rbf_nn.Batch_Pred(X_tst)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = True
        else:
            y_pred[i] = False
        if y_pred[i] == y_tst[i]:
            count += 1
    tst_err = 1 - count/float(len(y_tst))
    print "RBF network trained by accumulative BP method test error rate: %.3f" % \
    tst_err
    
    del err, err_k, count, i