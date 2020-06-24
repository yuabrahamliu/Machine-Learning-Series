# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:51:34 2018

@author: Yu Liu
"""

#%%
import numpy as np

def mean_vector(X, Y):
    """
    @param X: sample matrix (one row is one sample vector, only including \
                             sample attributes and not including 1 \
                             corresponding to intercept term)
    @param Y: label vector (one element represents the label of one sample in \
                            X)
    @return: mu (mean matrix, whose row number is the sample class number and \
                 each row represents the mean vector of one class. Each \
                 element of the row is the mean value of a specific attribute \
                 of the sample class)
    """
    mu = []
    for i in range(2):
        mu.append(np.mean(X[Y == i], axis = 0)) #column mean
    mu = np.array(mu)
    return mu

def within_class_scatter(X, Y):
    """
    Sw = sigma0 + sigma1 
       = sigma(x0i - mu0).T(x0i - m0) + sigma(x1i - mu1).T(x1i - mu1)
    
    @param X: sample matrix (one row is one sample vector, only including \
                             sample attributes and not including 1 \
                             corresponding to intercept term)
    @param Y: label vector (one element represents the label of one sample in \
                            X)
    @return: Sw matrix (the within class scatter matrix, 
                        a matrix, not a real number, and each element of the \
                        matrix is the sum of the covariance of different \
                        variables in the classes)
    """
    mu_matrix = mean_vector(X, Y)
    m,n = np.shape(X)
    Sw = np.zeros((n, n))
    for i in range(m):
        x = X[i]
        if Y[i] == 0:
            mu = mu_matrix[0]
        else:
            mu = mu_matrix[1]
        diff = x - mu
        diffT = diff.reshape(n, 1) #For a vector, use reshape to transform 
                                   #between row vector and column vector, 
                                   #not use .T, which is suitable for matrix
        Sw += np.dot(diffT, diff.reshape(1, n))
        #Before reshape, the shape of diff is (2,), without the column number 1, 
        #use reshape to set a column number to perform matrix mulplification
    return Sw

def orientation(X, Y):
    """
    w = Sw.inv(mu0 - mu1)
    
    @param X: sample matrix (one row is one sample vector, only including \
                             sample attributes and not including 1 \
                             corresponding to intercept term)
    @param Y: label vector (one element represents the label of one sample in \
                            X)
    @return: w (the unit vector representing orientation of the best \
                projection line)
    """
    Sw = within_class_scatter(X, Y)
    Sw = np.mat(Sw) #The former Sw is an array, change it to a matrix using 
                    #np.mat to perform singular value decomposition
    U,sigma,V = np.linalg.svd(Sw)
    Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
    #The former sigma is an array with the shape(2L, ), use np.diag to 
    #tranform it to a diagonal matrix
    
    m,n = np.shape(X)
    mu = mean_vector(X, Y)
    w = np.dot(Sw_inv, (mu[0] - mu[1]).reshape(n, 1))
    return w
    
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
    Y = data[:, 3]
    
    orientation(X, Y)
