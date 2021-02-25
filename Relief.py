# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:02:56 2021

@author: abrah
"""

#%%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

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
label = Series([1, 1, 1, 1, 1, 
                1, 1, 1, 0, 0, 
                0, 0, 0, 0, 0, 
                0, 0])

watermelon = pd.concat([Idx, color, root, knocks, texture, 
                        navel, touch, density, sugar_ratio, label], axis = 1)
#pd.concat(objs, axis = 0) Concatenate pandas objects along a particular axis 
#                          with optional set logic along the other axes. 
#objs: a sequence or mapping of Series, DataFrame, or Panel objects
#axis: The axis to concatenate along

watermelon.columns = ['Idx', 'color', 'root', 'knocks', 'texture', 
                      'navel', 'touch', 'density', 'sugar_ratio', 'label']
#Use watermelon.columns to rename columns

watermelon = watermelon.set_index(['Idx'], drop = True)
#Set the DataFrame index (row labels) using one or more existing columns. By 
#default yields a new object.

watermelon['label'] = watermelon['label'].map({1:'good', 0:'bad'})
#map(function, sequence) Return a list of the results of applying the funciton 
#                        to the ITEMs of the argument sequence

del Idx, color, root, knocks, texture, navel, touch, density, sugar_ratio, label

#%%
def Relief(X): 
    """
    Perform Relief algorithm to select features
    @X: DataFrame, each row is a sample and each column is a feature, except the 
        last column, which is the sample labels.
    """
    
    m = X.shape[0]     #Sample number
    n = X.shape[1] - 1 #Feature number
    
    X.index = range(m) #Ensure the index of the DataFrame X is 0-based like most 
                       #Python objects, so that the possiblity of meeting some 
                       #errors about index calling could be reduced.
    
    Y = X.ix[:,n]      #Sample labels, a Series
    X = X.ix[:,0:n]
        
    types = np.array([type(xj) for xj in X.ix[0,]])
    #Record the data type of each feature
    
    d_index = np.where(types == str)[0]
    #The column indeces of discrete features
    
    #np.where(condition, [x, y]) Return elements, either from `x` or `y`, depending 
    #  on `condition`.
    #  If only `condition` is given, return ``condition.nonzero()``.
    #  condition: array_like, bool. When True, yield `x`, otherwise yield `y`.
    #  x, y: array_like, optional. Values from which to choose. `x` and `y` need 
    #    to have the same shape as `condition`.
    #  out: ndarray or tuple of ndarrays. If both `x` and `y` are specified, the 
    #    output array contains elements of `x` where `condition` is True, and 
    #    elements from `y` elsewhere.
    #    If only `condition` is given, return the tuple ``condition.nonzero()``, 
    #    the indeces where `condition` is True.
    
    #np.nonzero(a) Return the indeces of the elements that are non-zero.
    #  Returns a tuple of arrays, one for each dimension of `a`, containing the 
    #  indeces of the non-zero elements in that dimension.
    #  a: array-like. Input array.    
    
    c_index = np.where(np.array([(xj == np.float64) | (xj == np.float32) | 
            (xj == np.float16) | (xj == np.float) | (xj == np.int) | 
            (xj == int) | (xj == float) for xj in types]))[0]
    #The column indeces of continuous features
    
    #tst is array([<type 'numpy.float64'>, <type 'numpy.float64'>], dtype=object)
    #For ``tst == float``, it returns 
    #  array([False, False], dtype=bool)
    #For ``tst[0] == np.float64``, it returns 
    #  True
    #However, for ``tst == np.float64``, it returns 
    #  False
    #Hence, in the case that tst is an array or a list, even if it only contains 
    #one element, ``tst == np.float64`` does not performs in an elementwise manner, 
    #which is different from ``tst == float``, and the result it returns is not 
    #the logistic judgement for the first element of the array or the list either, 
    #i.e. ``tst[0] == np.float64``
    #To get the array containing the elementwise logistic judgements for np.float64, 
    #instead of using ``tst == np.float64``, should use a comphrehension, 
    #that is ``np.array([xj == np.float64 for xj in tst])``
    
    #np.where(condition, [x, y]) Return elements, either from `x` or `y`, depending 
    #  on `condition`.
    #  If only `condition` is given, return ``condition.nonzero()``.
    #  condition: array_like, bool. When True, yield `x`, otherwise yield `y`.
    #  x, y: array_like, optional. Values from which to choose. `x` and `y` need 
    #    to have the same shape as `condition`. 
    #  out: ndarray or tuple of ndarrays. If both `x` and `y` are specified, the 
    #    output array contains elements of `x` where `condition` is True, and 
    #    elements from `y` elsewhere. 
    #    If only `condition` is given, return the tuple ``condition.nonzero()``, 
    #    the indeces where `condition` is True.
    
    #np.nonzero(a) Return the indeces of the elements that are non-zero.
    #  Returns a tuple of arrays, one for each dimension of `a`, containing the 
    #  indeces of the non-zero elements in that dimension. 
    #  a: array-like. Input array.
    
    Xd = X.ix[:,d_index] #The discrete part of X
    Xc = X.ix[:,c_index] #The continuous part of X
    
    def normalize(col): 
        normcol = (col - min(col))/(max(col) - min(col))
        
        return normcol
    
    Xc = pd.DataFrame.apply(Xc, normalize, axis = 0)
    #Scale the continuous features to the range [0, 1]
    
    #pd.DataFrame.apply(self, func, axis = 0) Applies function along input axis 
    #  of DataFrame.
    
    r = np.zeros(n) #Use to record the statistic for each feature
    for i in range(m): 
        dist2 = (Xd.ix[i,:] != Xd).sum(1) + ((Xc.ix[i,:] - Xc)**2).sum(1)
        #Calculate the distance square between the i-th sample and all samples, 
        #contributed by both the discrete features and continuous features
        
        #pd.DataFrame.sum(self, axis, skipna = None, level = None) Return the 
        #  sum of the values for the requested axis
        #  skipna: boolean, default True. Exclude NA/null values. If an entire 
        #    row/column is NA, the result will be NA.
        #  level: int or level name, default None. If the axis is a MultiIndex 
        #    (hierarchical), count along a particular level, collapsing into a 
        #    Series
        #  Returns: sum: Series or DataFrame (if level specified)
        
        #Find the near hit (nh) neighbor
        dist2_nh = dist2.copy()
        dist2_nh.ix[i] = max(dist2) + 1
        #The same type samples also include the i-th sample itself, to avoid 
        #selecting itself as the nearest same type neighbor, change the self 
        #distance from 0 to max(dist2) + 1
        dist2_nh.ix[Y != Y.ix[i]] = max(dist2) + 1
        #Also set the distance to the different type samples as max(dist2) + 1
        
        nh_index = np.argmin(dist2_nh)
        r[d_index] -= Xd.ix[i,:] != Xd.ix[nh_index,:]
        r[c_index] -= (Xc.ix[i,:] - Xc.ix[nh_index,:])**2
        
        #Find the near missing (nm) neighbor
        dist2_nm = dist2.copy()
        dist2_nm.ix[Y == Y.ix[i]] = max(dist2) + 1
        #Set the distance to the same type samples as max(dist2) + 1
        #Because the different type samples don't include the i-th sample itself, 
        #no need to concern the problem of self selection as processing the 
        #same type samples before
        
        nm_index = np.argmin(dist2_nm)
        r[d_index] += Xd.ix[i,:] != Xd.ix[nm_index,:]
        r[c_index] += (Xc.ix[i,:] - Xc.ix[nm_index,:])**2
        
    r = Series(r)
    r.name = 'statistic'
    #Use r.name to rename the Series r
    
    r = pd.concat([Series(X.columns), r], axis = 1)
    #pd.concat(objs, axis = 0) Concatenate pandas objects along a particular axis 
    #  with optional set logic along the other axes. 
    #  objs: a sequence or mapping of Series, DataFrame, or Panel objects 
    #  axis: The axis to concatenate along

    r = r.rename(columns = {0: 'feature'})
    #pd.DataFrame.rename(self, index = None, columns = None) Alter axes input 
    #  function or functions. Function/dict values must be unique (1-to-1). 
    #  Can change ``Series.name`` with a scalar value (Series only)
    #  index, columns: scalar, list-like, dict-like or function, optional. 
    #    Scalar or list-like will alter the ``Series.name`` attribute, and raise 
    #    on DataFrame or Panel. For Series, if set `INDEX` as a scalar, it will 
    #    change the colname of the Series
    #    dict-like or functions are transformations to apply to that axis' values. 
    #    For Series, if set `index` as a dict-like or function, it will change 
    #    the rowname of the Series
    
    r = r.sort_values(by = ['statistic'], axis = 0, ascending = False)
    #pd.DataFrame.sort_values(self, by, axis = 0, ascending = True) Sort by the 
    #  values along either axis
    #  by: str or list of str. Name or list of names which refer to the axis items. 
    #  ascending: bool or list of bool, default True. Sort ascending vs. descending. 
    #    Specify list for multiple sort orders. If this is a list of bools, must 
    #    match the length of the by.    
    
    r.index = range(n)
    #Use r.index to rename the index of r
    
    r = r.set_index(['feature'], drop = True)
    #Set the DataFrame index (row labels) using one or more existing columns. By 
    #default yields a new object.
        
    return r

#%%
if __name__ == '__main__': 
    r = Relief(X = watermelon)
    
    for i in range(r.shape[0]):
        print r.index[i] + ': ' + str(round(r.ix[i,0], 2))
        #round(number[, digits]) Round a number to a given precision in decimal 
        #  digits (default 0 digits). This always returns a floating point 
        #  number.
        
#%%
%reset
    
