# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:07:18 2019

@author: abrah
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV #For optimal parameter search

density = Series([0.697, 0.774, 0.634, 0.608, 0.556, 
                  0.403, 0.481, 0.437, 0.666, 0.243, 
                  0.245, 0.343, 0.639, 0.657, 0.360, 
                  0.593, 0.719])

sugar_content = Series([0.460, 0.376, 0.264, 0.318, 0.215, 
                        0.237, 0.149, 0.211, 0.091, 0.267, 
                        0.057, 0.099, 0.161, 0.198, 0.370, 
                        0.042, 0.103])

Id = Series(range(1, 18))

df = pd.concat([Id, sugar_content, density], axis = 1)
#pd.concat
#Concatenate pandas objects along a particular axis with optional set logic 
#along the other axes.
#axis, The axis to concatenate along

del Id, sugar_content, density
#Directly use del to remove objects

df.columns = ['Id', 'sugar_content', 'density']
#Directly use df.columnS to set column names

df = df.set_index(['Id'], drop = True)
#Set the DataFrame index (row labels) using one or more existing columns.
#By default yields a new object

df = df[['density', 'sugar_content']]
#Use df[[new column order]] to change the column order

np.corrcoef(df['density'], df['sugar_content'])
#Use np.corrcoef to get the correlation matrix

np.cov(df['density'], df['sugar_content'])
#Use np.cov to get the covariance matrix

pearsonr(df['density'], df['sugar_content'])
#Use pearsonr to calculate Pearson corelation coefficient 
#AND the p-value for testing non-correlation will ALSO be returned, like
#(0.19588910117802374, 0.45115105633434627)

X = df[['density']].get_values()
#Use get_values() to reduce the DataFrame to array
#Use df[['density']] to select a DataFrame with a col number of 1
#Use df['density'] to select a Series, with no col

y = df[['sugar_content']].get_values()
#Use get_values() to reduce the DataFrame to array
#Use df[['sugar_content']] to select a DataFrame with a col number of 1
#Use df['sugar_content'] to select a Series with no col

#Generate model and fitting###
for fignum, kernel in enumerate(('linear', 'rbf')):
    svr = GridSearchCV(SVR(kernel=kernel), cv=5, 
                           param_grid={"C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 
                                             1e1, 1e2, 1e3, 1e4]})
    #GridSearchCV
    #Exhaustive search over speicified parameter values for an estimator.
    #Important member are fit, predict.
    #GridSearchCV implements a "fit" and a "score" method
    #param_grid: dict or list of dictionaries
    #Dictionary with parameters names (string) as keys and lists of parameter 
    #settings to try as values, or a list of such dictionaries. This enables 
    #searching over any sequence of parameter settings.
    
    #scoring: string, callable or None, default = None. A string or a scorer 
    #callable object / function with signature ``scorer(estimator, X, y)``.
    #If ``None``, the ``score`` method of the estimator is used.
    #tst = SVR(kernel=kernel)
    #tst.score
    #Returns the coefficient of determination R^2 of the prediction
    #The coefficient R^2 is defined as (1 - u/v), where u is the regression sum 
    #of squares ((y_true - y_pred)**2).sum() and v is the residual sum of 
    #squares ((y_true - y_true.mean())**2).sum().
    #Best possible score is 1.0 and it can be negative (because the model can 
    #be arbitrarily worse).
    
    svr.fit(X, y)
    
    sv_ratio = float(svr.best_estimator_.support_.shape[0])/len(X)
    print("Support vector ratio: %.3f" % sv_ratio)
    
    y_svr = svr.predict(X)
    
    plt.figure(fignum)
    #Creates a new figure
    plt.clf()
    #Clear the current figure
    
    sv_ind = svr.best_estimator_.support_
    plt.scatter(X[sv_ind], y[sv_ind], c = 'r', s = 50, 
                label = 'SVR support vectors', zorder = 2)
    #Marker color is mapped to `c`
    #c: color, sequence, or sequence of color
    #Marker size is scaled by `s`
    #s: size in points^2.
    plt.scatter(X, y, c = 'k', label = 'data', zorder = 1)
    plt.plot(X, y_svr, c = 'orange', 
             label = "SVR fit curve with %s kernel" % kernel)
    plt.xlabel('density')
    plt.ylabel('sugar_ratio')
    plt.title('SVR on watermelon3.0a')
    plt.legend()

plt.show()
    










