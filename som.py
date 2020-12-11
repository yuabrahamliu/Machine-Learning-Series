# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:46:06 2018

@author: Yu Liu
"""

#%%
import numpy as np
import random
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

def distance(w, x):
    """
    Compute distance between array w and vector x
    
    @param: w, np.array with shape [m, d]
    @param: x, np.array with shape [1, d]
    @return: dist_square, np.array with shape [m, 1]
    """
    m = w.shape[0]
    if m == 1:
        dist_square = np.sum((w - x)**2)
    else:
        dist_square = np.sum((w - x)**2, axis = 1, keepdims = True)
    
    return dist_square

class SOMNet():
    
    def __init__(self, input_attr_num, output_num, 
                 sigma0, eta0, tau_sigma, tau_eta, iterations):
        """
        Construction method of the SOMNet class
        """
        self.input_attr_num = input_attr_num
        self.output_num = output_num
        self.sigma0 = sigma0
        self.eta0 = eta0
        self.tau_sigma = tau_sigma
        self.tau_eta = tau_eta
        self.iterations = iterations
        self.weights = np.random.rand(output_num, input_attr_num)
        #?np.random.rand
        #Create an array of the given shape and populate it with 
        #random samples from a uniform distribution over [0, 1)
        
    def update(self, x, iter):
        """
        sigma(t) = sigma0*exp(-t/tau_sigma)
        eta(t) = eta0*exp(-t/tau_eta)
        T_w_x = exp(-distance_square(w_x)/2*sigma(t)^2)
        delta_w = eta(t)*T_w_x*(x - w)
        
        @param: x, np.array with shape [1, d]. The current input vector 
        (one sample). d is the input attribute num.
        @param: iter, the current iteration number, int.
        """
        self.sigma = self.sigma0*np.exp(-iter/self.tau_sigma)
        self.eta = self.eta0*np.exp(-iter/self.tau_eta)
        neighbor_function = \
        np.exp(-distance(self.weights, x)/(2*self.sigma*self.sigma))
        #The col num of self.weights is the num of input attribute num
        #The row num of self.weights is the num of output neurons
        #The function distance will calculate the distance between each neuron 
        #and the input vector x (one sample)
        #Here, for one neuron, becaue the element num of its weight vector is 
        #equal to the num of input attribute num (the dimesion num of the input), 
        #only different output neurons arranged in one dimension line can 
        #complete the comparison between neurons and input data, this is a 
        #dimension reduction process for the original data. If the length of 
        #the output neuron vector is less than the num of input attribute, 
        #but not equal to 1, the dimension can also be reduced, but the output
        #neurons need to arrange in multiple dimensions (in other words, a 
        #neuron needs multiple vectors, for example, vector in dimesion 1, 
        #vector in dimension 2, ...) to represent all the attributes of the 
        #input neurons, so the final dimension num is greater than 1. If the 
        #element num of the weight vector is only one, there is no dimension 
        #reduction effect. So the effect of dimension reduction of SOM network 
        #relies on the length of the weight vector of each output neuron.
        
        self.weights = \
        self.weights + self.eta*neighbor_function*(x - self.weights)
        
    def train(self, train_X):
        """
        Train the weight matrix of the output neurons
        
        @param: train_X, list with length n and element is np.array with shape 
        [1, d]. Training instances. Each element is one sample. d is the input 
        attrubute num
        """
        n = len(train_X)
        #num of samples in the input sample set
        for iter in range(self.iterations):
            #Choose instance from train set randomly
            x = train_X[random.randint(0, n-1)]
            #update weight vectors for all output neurons
            self.update(x, iter)
            
        print self.sigma
        
    def eval(self, x):
        """
        Compute the best match unit (BMU) index given input vector x
        
        @param: x, np.array with shape [1, d]
        @return: index, the index of BMU
        """
        dist_square = distance(self.weights, x)
        index = np.argmin(dist_square)
        
        return index
    
#%%
if __name__ == '__main__':
    #Prepare train data
    col1 = Series([0.697, 0.774, 0.634, 0.608, 0.556, 
            0.403, 0.481, 0.437, 0.666, 0.243, 
            0.245, 0.343, 0.639, 0.657, 0.360, 
            0.593, 0.719])
    col2= Series([0.460, 0.376, 0.264, 0.318, 0.215, 
           0.237, 0.149, 0.211, 0.091, 0.267, 
           0.057, 0.099, 0.161, 0.198, 0.370, 
           0.042, 0.103])
    col3 = Series([1, 1, 1, 1, 1, 
            1, 1, 1, 0, 0, 
            0, 0, 0, 0, 0, 
            0, 0])
    book = pd.concat([col1, col2, col3], axis = 1)
    
    train_X = []
    train_y = []
    
    X1 = book.values[:, 0]
    X2 = book.values[:, 1]
    for i in range(len(X1)):
        train_X.append(np.array([[X1[i], X2[i]]]))
    train_y = book.values[:, 2]
    
    del col1, col2, col3, book, X1, X2, i
    
    #Train SOM network
    output_num = 4
    sigma0 = 3
    eta0 = 0.1
    tau_sigma = 1
    tau_eta = 1
    iterations = 100
    som_net = SOMNet(input_attr_num=2, output_num=output_num, 
                     sigma0=sigma0, eta0=eta0, 
                     tau_sigma=tau_sigma, tau_eta=tau_eta, 
                     iterations=iterations)
    som_net.train(train_X)
    
    #Plot data in 2 dimension space
    left_top_count = 0
    left_bottom_count = 0
    right_top_count = 0
    right_bottom_count = 0
    for i in range(len(train_X)):
        bmu = som_net.eval(train_X[i])
        if train_y[i] == 1:
            style = 'bo'
        else:
            style = 'r+'
        print bmu
        if bmu == 0:
            plt.plot([1+left_top_count*0.03], [2], style)
            left_top_count += 1
        elif bmu == 1:
            plt.plot([2+right_top_count*0.03], [2], style)
            right_top_count += 1
        elif bmu == 2:
            plt.plot([1+left_bottom_count*0.03], [1], style)
            left_bottom_count += 1
        else:
            plt.plot([2+right_bottom_count*0.03], [1], style)
            right_bottom_count += 1
    
    plt.xlim([1, 3])
    plt.ylim([1, 3])
    plt.show()