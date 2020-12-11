# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:11:13 2019

@author: Yu Liu
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import Series, DataFrame

from scipy.stats.stats import pearsonr

from sklearn import svm

density = Series([0.697, 0.774, 0.634, 0.608, 0.556, 
                  0.403, 0.481, 0.437, 0.666, 0.243, 
                  0.245, 0.343, 0.639, 0.657, 0.360, 
                  0.593, 0.719])

sugar_content = Series([0.460, 0.376, 0.264, 0.318, 0.215, 
                        0.237, 0.149, 0.211, 0.091, 0.267, 
                        0.057, 0.099, 0.161, 0.198, 0.370, 
                        0.042, 0.103])

Id = Series(range(1, 18))

label = [1]*8
#Use [1]*8 to create a list with 8 repeated elements 1.

label.extend([0]*9)
label = Series(label)

df = pd.concat([Id, sugar_content, density, label], axis = 1)
#pd.concat
#Concatenate pandas objects along a particular axis with optional set logic 
#along the other axes.
#axis, The axis to concatenate along

del Id, density, sugar_content, label
#Directly use del to remove objects

df.columns = ['id', 'sugar_content', 'density', 'label']
#Directly use df.columns to set column names

df = df.set_index('id', drop = True)
#Set the DataFrame index (row labels) using one or more existing columns.
#By default yields a new object

df = df[['density', 'sugar_content', 'label']]
#Use df[[new column order]] to change the column order


np.corrcoef(df['density'], df['sugar_content'])
#Use np.corrcoef to get the correlation matrix

np.cov(df['density'], df['sugar_content'])
#Use np.cov to get the covariance matrix

pearsonr(df['density'], df['sugar_content'])
#Use pearsonr to calculate Pearson corelation coefficient 
#AND the p-value for testing non-correlation will ALSO be returned, like
#(0.19588910117802374, 0.45115105633434627)

X = df[['density', 'sugar_content']].get_values()
#Use get_values() to reduce the DataFrame to array!

y = df[['label']].get_values()
#Use get_values() to reduce the Series to array
#Use df[['label']] to select a DataFrame with col number of 1
#Use df['label'] to select a Series with no col


#SVM trainiing and comparison######
#Based on linear kernal as well as gaussian kernal

for fig_num, kernel in enumerate(('linear', 'rbf')):
    #enumerate return an enumerate object.
    #The enumerate object yields pairs containing a count 
    #(from start, which defaults to zero) and a value yielded by the iterable 
    #argument.
    #enumerate is useful for obtaining an indexed list:
        #(0, seq[0]), (1, seq[1]), (2, seq[2]), ...
        
    #initial
    svc = svm.SVC(C = 1000, kernel = kernel)
    #C-Support Vector Classification
    #C: Penalty parameter C of the error term (default = 1.0)
    
    #train
    svc.fit(X, y)
    
    #get support vectors
    sv = svc.support_vectors_
    
    #Coefficients of the support vector in the decision function 
    #(The Lagrange multiplier alpha)
    #w = Sigma(alpha_i*y_i*x_i)
    coef = svc.dual_coef_
    
    #draw decision zone###
    plt.figure(fig_num)
    #Creates a new figure
    plt.clf()
    #Clear the current figure
    
    #Plot point and mark out support vectors
    plt.scatter(X[:,0], X[:,1], edgecolors='k', 
                c=y, cmap=plt.cm.Paired, zorder=10)
    #Marker color is mapped to `c`
    #c: color, sequence, or sequence of color
    #cmap: A `~matplotlib.colors.Colormap` instance or registered name
    #`cmap` is only used if `c` is an array of floats.
    plt.scatter(sv[:,0], sv[:,1], edgecolors='k', facecolors='none', 
                s=80, linewidths=2, zorder=10)
    #Marker size is scaled by `s`
    #s: size in points^2.
    
    #Plot the decision boundary and decision zone into a color plot
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), 
                         np.arange(y_min, y_max, 0.02))
    #np.arange 
    #Return evenly spaced values within a given interval
    #np.meshgrid
    #Return coordinated matrices from coordinated vectors
    
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()])
    #a.ravel([order])
    #Return a flattened array
    #np.c_
    #Translate slice objects to concatenation along the second axis
    
    #XX.ravel().shape
    #(1927L,)
    #YY.ravel().shape
    #(1927L,)
    #np.c_[XX.ravel(), YY.ravle()].shape
    #(1927L, 2L)
    
    #svc.decision_function
    #Distance of the samples X to the separating hyperplane
    #X: array-like, shape (n_samples, n_features)
    
    Z = Z.reshape(XX.shape)
    
    plt.pcolormesh(XX, YY, Z > 0, cmap = plt.cm.Paired)
    #Plot a quadrilateral mesh.
    plt.contour(XX, YY, Z, colors = ['k', 'k', 'k'], 
                linestypes = ['--', '-', '--'], levels = [-.5, 0, .5])
    
    plt.title(kernel)
    plt.axis('tight')
    
plt.show()

 
    
    