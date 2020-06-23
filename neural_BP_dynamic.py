# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:40:19 2018

@author: Yu Liu
"""

#%%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from random import random
from math import exp
from math import tanh

def rand(a, b):
    """
    random value generated for parameter initialization
    Theta form of Lagrange's mean value theorem
    
    @param: a, b, the upper and lower limits of the random value
    @return: xi, random value
    """
    theta = random()
    xi = (b - a)*theta + a
    return xi

def Sigmoid(x):
    """
    y = 1/(1 + exp(-x))
    
    @param: x, auto-variable
    @return: s, Sigmoid function output value
    """
    s = 1.0/(1.0 + exp(-x))
    return s

def SigmoidDerivate(y):
    """
    derivate = y*(1-y)
    
    @param: y, output value of Sigmoid function
    @return: sdv, Sigmoid derivate output value
    """
    sdv = y*(1-y)
    return sdv

def Tanh(x):
    """
    Calcualte the Tanh function output value
    
    @param: x, auto-variable
    @return: t, Tanh function output value
    """
    t = tanh(x)
    return t

def TanhDerivate(y):
    """
    derivate = 1-y*y
    
    @param: y, output value of Tanh function
    @return: tdv, Tanh derivate output value
    """
    tdv = 1-y*y
    return tdv

class BP_network:
    
    def __init__(self):
        """
        Construction method of BP_network class
        """
        #node numbers of the input, hidden and output layers
        self.i_n = 0
        self.h_n = 0
        self.o_n = 0
        
        #output value of the layers
        self.i_v = []
        self.h_v = []
        self.o_v = []
        
        #parameters (w, t)
        self.ih_w = [] #weights between input and hidden layers
        self.ho_w = [] #weights between hidden and output layers
        self.h_t = [] #threshold of the hidden layer nodes
        self.o_t = [] #threshold of the output layer nodes
        
        #activation function to activate the hidden and output layers 
        #and corresponding derivates for calcualting the gradient of 
        #hidden (eh) and output (gj) layers
        self.fun = {
                'Sigmoid': Sigmoid, 
                'SigmoidDerivate': SigmoidDerivate, 
                'Tanh': Tanh, 
                'TanhDerivate': TanhDerivate
                }
        
        #learning rates of hidden (eta2) and output (eta1) layers
        self.lr1 = []
        self.lr2 = []
        
    def CreateNN(self, ni, nh, no, actfun, learningrate1, learningrate2):
        """
        Build a BP network constructure and initialize the parameters
        
        @param: ni, nh, no, node numbers of the input, hidden and output layers
        @param: actfun, string, name of the activation function
        @param: learningrate1, learningrate2, learing rate of output and 
                hidden layers
        """
        #Set node numbers
        self.i_n = ni
        self.h_n = nh
        self.o_n = no
        
        #initialize output values of each layer
        self.i_v = np.zeros(self.i_n)
        self.h_v = np.zeros(self.h_n)
        self.o_v = np.zeros(self.o_n)
        
        #initialize weight of each connection (random initialization)
        self.ih_w = np.zeros([self.i_n, self.h_n])
        for i in range(self.i_n):
            for h in range(self.h_n):
                self.ih_w[i][h] = rand(0, 1)
        self.ho_w = np.zeros([self.h_n, self.o_n])
        for h in range(self.h_n):
            for j in range(self.o_n):
                self.ho_w[h][j] = rand(0, 1)
        self.h_t = np.zeros(self.h_n)
        for h in range(self.h_n):
            self.h_t[h] = rand(0, 1)
        self.o_t = np.zeros(self.o_n)
        for j in range(self.o_n):
            self.o_t[j] = rand(0, 1)
        
        #Choose activation function and its derivate
        self.af = self.fun[actfun]
        self.afd = self.fun[actfun+'Derivate']
        
        #Initialize learning rate
        self.lr1 = np.ones(self.o_n)*learningrate1
        self.lr2 = np.ones(self.h_n)*learningrate2
        
    def Pred(self, x):
        """
        y = f(sigma(wx) - theta)
        
        @param: x, the input array for input layer (one sample, array). Sample 
        attributes served as input array, each attribute value VECTOR element 
        after dummy matrix transformation corresponds to an array element
        """
        #Activate input layer (just directly transfer the input array x)
        for i in range(self.i_n):
            self.i_v[i] = x[i]
        
        #Activate hidden layer
        for h in range(self.h_n):
            total = 0
            for i in range(self.i_n):
                total += self.ih_w[i][h]*self.i_v[i]
            self.h_v[h] = self.af(total - self.h_t[h])
            
        #Activate output layer
        for j in range(self.o_n):
            total = 0
            for h in range(self.h_n):
                total += self.ho_w[h][j]*self.h_v[h]
            self.o_v[j] = self.af(total - self.o_t[j])
    
    def BackPropagate(self, x, y, method = 'standard'):
        """
        gj = (yj(k) - yj(k)hat)*f_dev(yj(k)hat)
        [For Sigmoid function, 
        f_dev(yj(k)hat) = yj(k)hat*(1-yj(k)hat)]
        [For Tanh function, 
        f_dev(yj(k)hat) = 1 - yj(k)hat*yj(k)hat]
        
        eh = sigma(whj*gj)*f_dev(bh)
        [For Sigmoid function, 
        f_dev(bh) = bh*(1-bh)]
        [For Tanh function, 
        f_dev(bh) = 1 - bh*bh]
        
        delta_whj = eta*gj*bh
        delta_thetaj = -eta*gj
        
        delta_vih = eta*eh*xi
        delta_gammah = -eta*eh
        
        @param: x, the input array for input layer (one sample, array). Sample 
        attributes served as input array, each attribute value VECTOR element 
        after dummy matrix transformation corresponds to an array element
        @param: y, sample label served as output array (one sample, array), 
        each label value VECTOR element after dummy matrix transformation 
        corresponds to an array element
        @param: method, string, type of BP algorithsm to be used, 'standard' or 
        'accumulative'
        """
        #Get current network output
        self.Pred(x)
        
        #Calculate the gradients based on output values of output layer and 
        #hidden layer
        o_grid = np.zeros(self.o_n)
        for j in range(self.o_n):
            o_grid[j] = (y[j] - self.o_v[j])*self.afd(self.o_v[j])
            
        h_grid = np.zeros(self.h_n)
        for h in range(self.h_n):
            for j in range(self.o_n):
                h_grid[h] += self.ho_w[h][j]*o_grid[j]
            h_grid[h] = h_grid[h]*self.afd(self.h_v[h])
        
        #Update the parameters
        if method == 'accumulative':
            self.delta_ho_w = np.zeros([self.h_n, self.o_n])
            self.delta_ih_w = np.zeros([self.i_n, self.h_n])
            self.delta_o_t = np.zeros(self.o_n)
            self.delta_h_t = np.zeros(self.h_n)
            
        for h in range(self.h_n):
            for j in range(self.o_n):
                if method == 'standard':
                    self.ho_w[h][j] += self.lr1[j]*o_grid[j]*self.h_v[h]
                else:
                    self.delta_ho_w[h][j] += \
                    self.lr1[j]*o_grid[j]*self.h_v[h]
                
        for i in range(self.i_n):
            for h in range(self.h_n):
                if method == 'standard':
                    self.ih_w[i][h] += self.lr2[h]*h_grid[h]*self.i_v[i]
                else:
                    self.delta_ih_w[i][h] += \
                    self.lr2[h]*h_grid[h]*self.i_v[i]
                
        for j in range(self.o_n):
            if method == 'standard':
                self.o_t[j] -= self.lr1[j]*o_grid[j]
            else:
                self.delta_o_t[j] -= self.lr1[j]*o_grid[j]
                
        for h in range(self.h_n):
            if method == 'standard':
                self.h_t[h] -= self.lr2[h]*h_grid[h]
            else:
                self.delta_h_t[h] -= self.lr2[h]*h_grid[h]
    
    def TrainData(self, data_in, data_out, method = 'standard'):
        """
        Perform standard or accumulative BP training
        
        @param: data_in, Sample attribute data set (array). Each row 
        corresponds to a sample. Each attribute value VECTOR element after 
        dummy matrix transformation corresponds to a column
        @param: data_out, Sample label data set (array). Each row corresponds 
        to a sample. Each label value VECTOR element after dummy matrix 
        transformation corresponds to a column
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
            self.BackPropagate(x, y, method=method)
            
            #Error in train set for each step
            y_delta2 = 0.0
            for j in range(self.o_n):
                delta_yj_2 = (self.o_v[j] - y[j])*(self.o_v[j] - y[j])
                #The error of the jth element of the label vector of one sample
                y_delta2 += delta_yj_2
                #y_delta2 is the sum of error of all elements in the label 
                #vector of one sample
            e_k.append(y_delta2/2.0)
            
        #Total error of all the samples
        e = sum(e_k)/len(e_k)
        
        if method == 'accumulative':
            self.ho_w += self.delta_ho_w/samplesize
            self.ih_w += self.delta_ih_w/samplesize
            self.o_t += self.delta_o_t/samplesize
            self.h_t += self.delta_h_t/samplesize
            
        return e, e_k
    
    def BackPropagate_Dynamic_Lr(self, x, y, 
                                 d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, 
                                 o_grid_p, h_grid_p, 
                                 alpha, 
                                 method = 'standard'):
        """
        gj = (yj(k) - yj(k)hat)*f_dev(yj(k)hat)
        [For Sigmoid function, 
        f_dev(yj(k)hat) = yj(k)hat*(1-yj(k)hat)]
        [For Tanh function, 
        f_dev(yj(k)hat) = 1 - yj(k)hat*yj(k)hat]
        
        eh = sigma(whj*gj)*f_dev(bh)
        [For Sigmoid function, 
        f_dev(bh) = bh*(1-bh)]
        [For Tanh function, 
        f_dev(bh) = 1 - bh*bh]
        
        delta_whj(t) = eta1*gj(t)*bh + alpha*delta_whj(t-1)
        delta_thetaj(t) = -eta1*gj(t) + alpha*delta_thetaj(t-1)
        delta_vih(t) = eta2*eh(t)*xi + alpha*delta_vih(t-1)
        delta_gammah(t) = -eta2*eh(t) + alpha*delta_gammah(t-1)
        
        lambda1 = sign[gj(t)*gj(t-1)]
        eta1(t) = 2^lambda1*eta1(t-1)
        
        lambda2 = sign[eh(t)eh(t-1)]
        eta2(t) = 2^lambda2*eta2(t-1)
        
        @param: x, the input array for input layer (one sample, array). Sample 
        attributes served as input array, each attribute value VECTOR element 
        after dummy matrix transformation corresponds to an array element
        @param: y, sample label served as output array (one sample, array), 
        each label value VECTOR element after dummy matrix transformation 
        corresponds to an array element
        @param: method, string, type of BP algorithsm to be used, 'standard' or 
        'accumulative'
        @param: d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, ajusted values (delta) of 
        last step
        @param: o_grid_p, h_grid_p, gradient of last step
        @param: alpha, forget factor
        @return: d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, o_grid_p, h_grid_p
        """
        #get current network output
        self.Pred(x)
        
        #Calcualte the gradients based on output values of output layer and 
        #hidden layer
        o_grid = np.zeros(self.o_n)
        for j in range(self.o_n):
            o_grid[j] = (y[j] - self.o_v[j])*self.afd(self.o_v[j])
            
        h_grid = np.zeros(self.h_n)
        
        for h in range(self.h_n):
            for j in range(self.o_n):
                h_grid[h] += self.ho_w[h][j]*o_grid[j]
            h_grid[h] = h_grid[h]*self.afd(self.h_v[h])
        
        #Update the parameters
        if method == 'accumulative':
            self.delta_ho_w = np.zeros([self.h_n, self.o_n])
            self.delta_ih_w = np.zeros([self.i_n, self.h_n])
            self.delta_o_t = np.zeros([self.o_n])
            self.delta_h_t = np.zeros([self.h_n])
            
        Lambda = np.sign(o_grid*o_grid_p)
        #Lambda is a 1-dimensional array with the same length of o_grid, 
        #so each node in the output layer has its specific Lambda, not 
        #all the output layer nodes share a same Lambda
        o_grid_p = o_grid
        for h in range(self.h_n):
            for j in range(self.o_n):
                #update learning rate
                lr = (2**Lambda[j])*self.lr1[j]
                if lr > 0.5:
                    self.lr1[j] = 0.5
                elif lr < 0.005:
                    self.lr1[j] = 0.005
                else:
                    self.lr1[j] = lr
                #update connection weights
                d_ho_w_p[h][j] = self.lr1[j]*o_grid[j]*self.h_v[h] + \
                alpha*d_ho_w_p[h][j]
                if method == 'standard':
                    self.ho_w[h][j] += d_ho_w_p[h][j]
                else:
                    self.delta_ho_w[h][j] += d_ho_w_p[h][j]
                    
        Lambda = np.sign(h_grid*h_grid_p)
        #Lambda is a 1-dimensional array with the same length of h_grid,
        #so each node in the hidden layer has its specific Lambda, not 
        #all the hidden layer nodes share a same Lambda
        h_grid_p = h_grid
        for i in range(self.i_n):
            for h in range(self.h_n):
                #update learning rate
                lr = (2**Lambda[h])*self.lr2[h]
                if lr > 0.5:
                    self.lr2[h] = 0.5
                elif lr < 0.005:
                    self.lr2[h] = 0.005
                else:
                    self.lr2[h] = lr
                #update connection weights
                d_ih_w_p[i][h] = self.lr2[h]*h_grid[h]*self.i_v[i] + \
                alpha*d_ih_w_p[i][h]
                if method == 'standard':
                    self.ih_w[i][h] += d_ih_w_p[i][h]
                else:
                    self.delta_ih_w[i][h] += d_ih_w_p[i][h]
        
        for j in range(self.o_n):
            d_o_t_p[j] = -(self.lr1[j]*o_grid[j] + alpha*d_o_t_p[j])
            if method == 'standard':
                self.o_t[j] += d_o_t_p[j]
            else:
                self.delta_o_t[j] += d_o_t_p[j]
        
        for h in range(self.h_n):
            d_h_t_p[h] = -(self.lr2[h]*h_grid[h] + alpha*d_h_t_p[h])
            if method == 'standard':
                self.h_t[h] += d_h_t_p[h]
            else:
                self.delta_h_t[h] += d_h_t_p[h]
        
        return d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, o_grid_p, h_grid_p
    
    def TrainData_Dynamic_Lr(self, data_in, data_out, method = 'standard'):
        """
        Perform standard or accumulative BP training
        
        @param: data_in, Sample attribute data set (array). Each row 
        corresponds to a sample. Each attribute value VECTOR element after 
        dummy matrix transformation corresponds to a column
        @param: data_out, Sample label data set (array). Each row corresponds 
        to a sample. Each label value VECTOR element after dummy matrix 
        transformation corresponds to a column
        @param: method, string, type of BP algorithsm to be used, 'standard' or 
        'accumulative'
        @return: e, accumulated error
        @return: e_k, error list consisting of the error of all the samples
        """
        d_ho_w_p = np.zeros([self.h_n, self.o_n])
        d_ih_w_p = np.zeros([self.i_n, self.h_n])
        d_o_t_p = np.zeros(self.o_n)
        d_h_t_p = np.zeros(self.h_n)
        o_grid_p = np.ones(self.o_n)*0.1
        h_grid_p = np.ones(self.h_n)*0.1
        
        e_k = []
        samplesize = len(data_in)
        for k in range(samplesize):
            x = data_in[k]
            y = data_out[k]
            d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, o_grid_p, h_grid_p = \
            self.BackPropagate_Dynamic_Lr(x, y, 
                                          d_ho_w_p, d_ih_w_p, d_o_t_p, d_h_t_p, 
                                          o_grid_p, h_grid_p, alpha = 0.2, 
                                          method = method)
            #Error in train set for each step
            y_delta2 = 0.0
            for j in range(self.o_n):
                delta_yj_2 = (self.o_v[j] - y[j])*(self.o_v[j] - y[j])
                #The error of the jth element of the label vector of one sample
                y_delta2 += delta_yj_2
                #y_delta2 is the sum of error of all elements in the label
            e_k.append(y_delta2/2.0)
            
        #Total error of all the samples
        e = sum(e_k)/len(e_k)
        
        if method == 'accumulative':
            self.ho_w += self.delta_ho_w/samplesize
            self.ih_w += self.delta_ih_w/samplesize
            self.h_t += self.delta_h_t/samplesize
            self.o_t += self.delta_o_t/samplesize
        
        return e, e_k
    
    def PredLabel(self, X):
        """
        Predict via the trained network
        
        @param: X, Sample attribute data set. Each row corresponds to a sample. 
        Each attribute value VECTOR element after dummy matrix transformation 
        corresponds to a column
        @return: y, Predicted sample label list. Each element corresponds to 
        the predicted class of a sample. For a sample, the single predicted 
        class value is chosen from all the possible classes by comparing the 
        predicted values for all the classes and the one with the maximum 
        value is used as final predicted class (winner-takes-all)
        """
        y = []
        for m in range(len(X)):
            self.Pred(X[m])
            max_y = self.o_v[0]
            label = 0
            for j in range(1, self.o_n):
                if max_y < self.o_v[j]:
                    label = j
                    max_y = self.o_v[j]
                    #Use the principle of bubble sorting
                    
            y.append(label)
            
        return np.array(y)
#%%
if __name__ == '__main__':
    
    Idx = Series(range(1, 18))
    color = Series(['dark_green', 'black', 'black', 'dark_green', 'light_white', 
                   'dark_green', 'black', 'black', 'black', 'dark_green', 
                   'light_white', 'light_white', 'dark_green', 'light_white', 
                   'black', 'light_white', 'dark_green'])
    root = Series(['curl_up', 'curl_up', 'curl_up', 'curl_up', 'curl_up', 
                   'little_curl_up', 'little_curl_up', 'little_curl_up', 
                   'little_curl_up', 'stiff', 
                   'stiff', 'curl_up', 'little_curl_up', 'little_curl_up', 
                   'little_curl_up', 'curl_up', 'curl_up'])
    knocks = Series(['little_heavily', 'heavily', 'little_heavily', 
                     'heavily', 'little_heavily', 
                     'little_heavily', 'little_heavily', 'little_heavily', 
                     'heavily', 'clear', 
                     'clear', 'little_heavily', 'little_heavily', 'heavily', 
                     'little_heavily', 'little_heavily', 'heavily'])
    texture = Series(['distinct', 'distinct', 'distinct', 'distinct', 'distinct', 
                      'distinct', 'little_blur', 'distinct', 'little_blur', 
                      'distinct', 
                      'blur', 'blur', 'little_blur', 'little_blur', 'distinct', 
                      'blur', 'little_blur'])
    navel = Series(['sinking', 'sinking', 'sinking', 'sinking', 'sinking', 
                    'little_sinking', 'little_sinking', 'little_sinking', 
                    'little_sinking', 'even', 
                    'even', 'even', 'sinking', 'sinking', 'little_sinking', 
                    'even', 'little_sinking',])
    touch = Series(['hard_smooth', 'hard_smooth', 'hard_smooth', 'hard_smooth', 
                    'hard_smooth', 
                    'soft_stick', 'soft_stick', 'hard_smooth', 'hard_smooth', 
                    'soft_stick',
                    'hard_smooth', 'soft_stick', 'hard_smooth', 'hard_smooth', 
                    'soft_stick', 
                    'hard_smooth', 'hard_smooth'])
    density = Series([0.697, 0.774, 0.634, 0.608, 0.556, 
                      0.403, 0.481, 0.437, 0.666, 0.243, 
                      0.245, 0.343, 0.639, 0.657, 0.36, 
                      0.593, 0.719])
    sugar_ratio = Series([0.46, 0.376, 0.264, 0.318, 0.215, 
                          0.237, 0.149, 0.211, 0.091, 0.267, 
                          0.057, 0.099, 0.161, 0.198, 0.37, 
                          0.042, 0.103])
    label = Series(['good', 'good', 'good', 'good', 'good', 
                    'good', 'good', 'good', 'bad', 'bad', 
                    'bad', 'bad', 'bad', 'bad', 'bad', 
                    'bad', 'bad'])
    watermelon = pd.concat([Idx, color, root, knocks, texture, navel, touch, 
                            density, sugar_ratio, label], axis = 1)
    watermelon.columns = ['Idx', 'color', 'root', 'knocks', 'texture', 
                           'navel', 'touch', 'density', 'sugar_ratio', 'label']
    watermelon = watermelon.set_index('Idx', drop = True)
    
    del Idx, color, root, knocks, texture, navel, touch, density, \
    sugar_ratio, label
    
    watermelon = pd.get_dummies(watermelon)
    
    test_index = [4, 8, 11, 13]
    test_data = watermelon.ix[test_index]
    train_data = watermelon[~watermelon.index.isin(test_index)]
    test_data = test_data.reset_index(drop = True)
    train_data = train_data.reset_index(drop = True)
    
    del test_index, watermelon
    
    df_test = test_data
    df_train = train_data
    
    del test_data, train_data
    
    #one-hot encoding
    X_test = df_test[df_test.columns[0:-2]].get_values()
    #Use get_values() to reduce the DataFrame to array!
    Y_test = df_test[df_test.columns[-2:]].get_values()
    label_test = pd.unique(df_test.columns)[-2:]
    
    X_train = df_train[df_train.columns[0:-2]].get_values()
    Y_train = df_train[df_train.columns[-2:]].get_values()
    label_train = pd.unique(df_train.columns)[-2:]
#%% Train a BP network using standard method
    bpn1 = BP_network() #Iinialize a BP network class
    bpn1.CreateNN(ni=int(X_train.shape[1]), nh=10, 
                  no=int(Y_train.shape[1]), actfun='Sigmoid', 
                  learningrate1=0.05, learningrate2=0.05)
    #Build the network
    
    e1 = []
    #Train the network using standard BP method
    for i in range(1000):
        err, err_k = bpn1.TrainData(data_in=X_train, data_out=Y_train, 
                                    method='standard')
        e1.append(err)
    
    #Get the test error in the test set
    pred1 = bpn1.PredLabel(X_test)
    count1 = 0
    for i in range(len(Y_test)):
        max_y = Y_test[i][0]
        label = 0
        for j in range(1, len(Y_test[i])):
            if max_y < Y_test[i][j]:
                max_y = Y_test[i][j]
                label = j
        if pred1[i] == label:
            count1 += 1
    
    test_err1 = 1 - count1/float(len(Y_test))
    print "Standard BP algorithsm test error rate: %.3f" % test_err1
    
    del err, err_k, count1, i, j, max_y, label
#%% Train a BP network using standard method with dynamic learning rates
    bpn1_d = BP_network() #Initialize a BP network class
    bpn1_d.CreateNN(ni=int(X_train.shape[1]), nh=10, 
                    no=int(Y_train.shape[1]), actfun='Sigmoid', 
                    learningrate1=0.05, learningrate2=0.05)
    #Build the network
    
    e1_d = []
    #Train the network using standard BP method with dynamic learning rates
    for i in range(1000):
        err, err_k = bpn1.TrainData_Dynamic_Lr(data_in=X_train, 
                                               data_out=Y_train, 
                                               method='standard')
        e1_d.append(err)
    
    #Get the test error in the test set
    pred1_d = bpn1_d.PredLabel(X_test)
    count1 = 0
    for i in range(len(Y_test)):
        max_y = Y_test[i][0]
        label = 0
        for j in range(1, len(Y_test[i])):
            if max_y < Y_test[i][j]:
                max_y = Y_test[i][j]
                label = j
        if pred1_d[i] == label:
            count1 += 1
            
    test_err1_d = 1 - count1/float(len(Y_test))
    print "Standard BP algorithsm with dynamic learning rates test error rate: %.3f"\
    % test_err1_d
    
    del err, err_k, count1, i, j, max_y, label
#%% Train a BP network using accumulative method
    bpn2 = BP_network() #Initialize a BP network class
    bpn2.CreateNN(ni=int(X_train.shape[1]), nh=10, 
                  no=int(Y_train.shape[1]), actfun='Sigmoid', 
                  learningrate1=0.05, learningrate2=0.05)
    #Build the network
    
    e2 = []
    #Train the network using accumulative BP method
    for i in range(1000):
        err, err_k = bpn2.TrainData(data_in=X_train, data_out=Y_train, 
                                    method='accumulative')
        e2.append(err)
    
    #Get the test error in test set
    pred2 = bpn2.PredLabel(X_test)
    count2 = 0
    for i in range(len(Y_test)):
        max_y = Y_test[i][0]
        label = 0
        for j in range(1, len(Y_test[i])):
            if max_y < Y_test[i][j]:
                max_y = Y_test[i][j]
                label = j
        if pred2[i] == label:
            count2 += 1
        
    test_err2 = 1 - count2/float(len(Y_test))
    print "Accumulative BP algorithsm test error rate: %.3f" % test_err2
    
    del err, err_k, count2, i, j, max_y, label
#%% Train a BP network using accumulative method with dynamic learning rates
    bpn2_d = BP_network() #Initialize a BP network class
    bpn2_d.CreateNN(ni=int(X_train.shape[1]), nh=10, 
                    no=int(Y_train.shape[1]), actfun='Sigmoid', 
                    learningrate1=0.05, learningrate2=0.05)
    #Build the network
    
    e2_d = []
    #Train the network using accumulative BP method with dynamic learning rates
    for i in range(1000):
        err, err_k = bpn2.TrainData_Dynamic_Lr(data_in=X_train, 
                                               data_out=Y_train, 
                                               method='accumulative')
        e2_d.append(err)
    
    #Get the test error in test set
    pred2_d = bpn2_d.PredLabel(X_test)
    count2 = 0
    for i in range(len(Y_test)):
        max_y = Y_test[i][0]
        label = 0
        for j in range(1, len(Y_test[i])):
            if max_y < Y_test[i][j]:
                max_y = Y_test[i][j]
                label = j
        if pred2_d[i] == label:
            count2 += 1
    
    test_err2_d = 1 - count2/float(len(Y_test))
    print "Accumulative BP algorithsm with dynamic learning rates test error rate: %.3f" \
    % test_err2_d
    
    del err, err_k, count2, i, j, max_y, label
    