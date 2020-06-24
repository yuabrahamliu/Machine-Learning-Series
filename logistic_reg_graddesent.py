# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:18:38 2018

@author: Yu Liu
"""

#%%
import numpy as np

def likelihood_sub(x, y, beta):
    """
    sub_log_likelihood = -y*beta*x.T + ln(1 + exp(beta*x.T))
    
    @param x: one sample variables (row vector, \
                                    including sample attributes and 1)
    @param y: one sample label, scalar
    @param beta: parameter vector (row vector, beta = (w, b))
    @return: sub_log_likelihood
    """
    sub_log_likelihood = -y*np.dot(beta, x.T) + \
                         np.math.log(1 + np.math.exp(np.dot(beta, x.T)))
    return sub_log_likelihood

def likelihood(X, Y, beta):
    """
    log_likelihood = sigma(sub_log_likelihood)
    
    @param X: sample variable matrix (one row is one sample vector, \
                                      including sample attributes and 1)
    @param Y: sample label vector (one scalar element is one sample label)
    @param beta: parameter vector (row vector, beta = (w, b))
    @return: log_likelihood
    """
    log_likelihood = 0
    m,n = np.shape(X)
    for i in range(m):
        x = X[i]
        y = Y[i]
        sub_log_likelihood = likelihood_sub(x, y, beta)
        log_likelihood += sub_log_likelihood
    return log_likelihood

def gradDsent(X, Y):
    """
    f(beta + delta_beta) ~ f(beta) + delta_beta*gradient
    delta_beta = -h*gradient
    
    @param X: sample variable matrix (one row is one sample vector, \
                                      including sample attributes and 1)
    @param Y: sample label vector (one scalar element is one sample)
    @return: the best parameter estimation of beta (beta = (w, b))
    """
    h = 0.1 # step length
    max_times = 500
    m,n = np.shape(X)
    
    beta = np.zeros(n) # Vector, one element corresponds to one attribute of 
                       # samples (beta = (w, b))
    delta_beta = np.ones(n)*h
    
    f_last = 0
    f_curr = 0
    
    for i in range(max_times):
        beta_temp = beta.copy()
        
        #for partial derivative
        for j in range(n):
            beta[j] += delta_beta[j]
            f_curr = likelihood(X, Y, beta)
            
            gradient = (f_curr - f_last)/delta_beta[j]
            delta_beta[j] = -h*gradient
            
            beta[j] = beta_temp[j]
        beta += delta_beta
        f_last = likelihood(X, Y, beta)
    return beta

def single_predict(x, beta):
    """
    y = 1/(1 + exp(-beta*x.T))
    
    @param x: the vector for one sample, each element corresponds to \
              an attribute or 1
    @param beta: parameter vector (beta = (w, b))
    @return: the predict value for one sample
    """
    y = 1.0/(1 + np.math.exp(-np.dot(beta, x.T)))
    return y

def logit_predict(X, beta):
    """
    @param X: sample matrix (one row is one sample vector, including sample \
                             attributes and 1)
    @param beta: parameter vector (beta = (w, b))
    @return: the logit regression result for all samples
    """
    m,n = np.shape(X)
    y = np.zeros(m)
    for i in range(m):
        if single_predict(X[i], beta) > 0.5:
            y[i] = 1
        else:
            y[i] = 0
    return y
    
#%%data
if __name__ == '__main__':
    data = np.array([[1, 0.697, 0.46, 1], 
                    [2, 0.774, 0.376, 1], 
                    [3, 0.634, 0.264, 1], 
                    [4, 0.608, 0.318, 1], 
                    [5, 0.556, 0.215, 1], 
                    [6, 0.403, 0.237, 1], 
                    [7, 0.481, 0.149, 1], 
                    [8, 0.437, 0.211, 1], 
                    [9, 0.666, 0.091, 0], 
                    [10, 0.243, 0.0267, 0], 
                    [11, 0.245, 0.057, 0], 
                    [12, 0.343, 0.099, 0], 
                    [13, 0.639, 0.161, 0], 
                    [14, 0.657, 0.198, 0], 
                    [15, 0.36, 0.37, 0], 
                    [16, 0.593, 0.042, 0], 
                    [17, 0.719, 0.103, 0]])
    X = data[:, 1:3]
    intercept = np.array([1]*17)
    X = np.column_stack((X, intercept.T))
    #Use np.column_stack to combine a matrix with a column vector
    Y = data[:, 3]
    
    beta = gradDsent(X, Y)
    result = logit_predict(X, beta)
    
    


