# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 02:59:37 2020

@author: abrah
"""

#%%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.utils.multiclass import type_of_target

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

watermelon = pd.concat([Idx, color, root, knocks, texture, navel, touch, 
                        density, sugar_ratio, label], axis = 1)
#pd.concat(objs, axis = 0) Concatenate pandas objects along a particular axis 
#                          with optional set logic along the other axes.
#objs: a sequence or mapping of Series, DataFrame, or Panel objects
#axis: The axis to concatenate along

watermelon.columns = ['Idx', 'color', 'root', 'knocks', 'texture', 
                       'navel', 'touch', 'density', 'sugar_ratio', 'label']
#Use watermelon.columns to rename colnames

watermelon = watermelon.set_index('Idx', drop = True)
#Set the DataFrame index (row labels) using one or more existing columns. By 
#default yields a new object.

watermelon['label'] = watermelon['label'].map({0: 'bad', 1: 'good'})
#map(function, sequence) Return a list of the results of applying the function 
#                        to the ITEMs of the argument sequence

del Idx, color, root, knocks, texture, navel, touch, density, sugar_ratio, label


test1 = pd.DataFrame([[1, 'dark_green', 'curl_up', 'little_heavily', 
                       'distinct', 'sinking', 'hard_smooth', 
                       0.697, 0.460, 'notknown']], \
                     columns = watermelon.columns.insert(0, 'idx'))
#Use watermelon.columns.insert(loc, item) to make new Index object with new 
#item inserted at location loc.
test1 = test1.set_index('idx', drop = True)
#Set the DataFrame index (row labels) using one or more existing columns. By 
#default yields a new object.

#%%
class AODE():
    def __init__(self, cutoff):
        """
        @cutoff: 
            The cuoff to select attribute levels with a sample number 
            greater than cutoff
        """
        self.m_hat = cutoff       #m_hat is the cutoff to select attribute levels 
                                  #with a sample number greater than cutoff
        self.m = None             #Sample number of the whole dataset
        self.n = None             #Attribute number of the whole dataset
        self.unique_y = None      #Class label levels
        self.n_class = None       #Number of the class label levels
        self.is_continuous = None #A Series with boolean values indicating if 
                                  #each attribute is continuous or not
        self.unique_values = None #A dictionary recording the levels of each 
                                  #discrete attribute. Each key corresponds to 
                                  #one discrete attribute and its value is the 
                                  #levels of that discrete attribute
        self.total_p = None       #A dictionary recording the probability 
                                  #of P(class label = c, parent attribute = xi) 
                                  #and P(attribute = xj | class label = c, 
                                  #parent attribute = xi). Each key corresponds 
                                  #to one parent attribute and its value is a 
                                  #DataFrame recording these probabilities using 
                                  #different class label levels as columns and 
                                  #different parent attribute levels as indexes, 
                                  #and each cell is a list containing these 
                                  #probablilities on various xj when xi and c 
                                  #are fixed.
    
    #During the process of writing a class, if want to test its methods being 
    #written, an annoyance is the error "NameError: name 'self' is not defined". 
    #To get a self object and use it conviently during method writing and testing, 
    #can write the __init__ method of the class first, and generate an object 
    #named 'self' using the temporary class containing the __init__ method only, 
    #like self = AODE(0) here, and then this self can be used to test other 
    #methods need to be written.
    
    def _get_parent_attribute(self, X):
        """
        Determine which discrete attribures can become parent attributes.
        If an attribute is a continuous attribute, it cannot become parent 
        attribute, because if use a continous attribute xj as a parent 
        attribute, it will be tiring to calculate the probability of 
        p(xi | c, xj) using Bayesian formula. In addition, AODE itself requires 
        to exclude attributes with a sample size less than the cutoff, from 
        parent attribute, so if think from this perspective, any continuous 
        attribute cannot be a parent attribute. Hence, no continous attribute 
        will be defined as a parent attribute here.
        @X
            DataFrame with only the attribute values (no class labels) of 
            the training dataset. Each row is a sample, while each column is 
            an attribute (no class label)
        """
        enough_quantity = pd.DataFrame(X).apply(
                lambda x: (type_of_target(list(x)) != 'continuous') & 
                (pd.value_counts(x) > self.m_hat).all()
                )
        #type_of_target(y) Determine the type of data indicated by target `y`
        #                  One of: 'continuous', 'binary', 'multiclass', 
        #                          'multiclass-mulitoutpout', 
        #                          'multilabel-indicator', and 'unknown'
        #If want to judge if the values are continuous or not, the values cannot 
        #be included in a Series, otherwise, even if the values are continuous, 
        #will still get 'unknown'. To judge continuous values, change the Series 
        #to a list
        #
        #pd.value_counts(values) Compute a histogram of the counts of non-null 
        #                        values
        #Returns: value_counts Series
        #
        #all(interable) -> bool Return True if bool(x) is True for all values 
        #                       x in the iterable
        
        return enough_quantity[enough_quantity].index.tolist()
    
    def _spode_fit(self, X, y, xi_index):
        """
        @X: 
            DataFrame with only the attribute values (no class labels) of 
            the training dataset. Each row is a sample, while each column is 
            an attribute (no class label)
        @y: 
            Series recording the label values of the training dataset.
        @xi_index: 
            ONE parent attribute name.        
        """
        p = pd.DataFrame(columns = self.unique_y, 
                         index = self.unique_values[xi_index])
        #Actually, unique_values was calculated within another method fit(), 
        #and if want to use it here, one way is to transfer it via the 
        #parameters of the current method _spode_fit(), which is generally used 
        #when writing an ordinary function (not method), while the other way 
        #is specific for the methods of a class, the object needs to be used 
        #(like unique_values here) can directly be transferred to the current 
        #method using self.unique_values, without specially transferring it 
        #using the parameters of current method.
        
        nunique_xi = self.unique_values[xi_index].size
        #Number of levels of the ONE parent attribute (discrete attribute)
        
        pc_xi_denominator = self.m + self.n_class*nunique_xi
        #The denominator of p(c, xi), which is |D| + N*N_i, where |D| is the 
        #total sample number, N is the total label level number, while 
        #N_i is the number of levels of ONE parent attribute in the WHOLE
        #dataset.
        
        for c in self.unique_y:
            #c is one level name of the label
            for xi in self.unique_values[xi_index]: 
                #xi is one level name of current parent attribute
                p_list = []
                c_xi = (X.ix[:,xi_index] == xi) & (y == c)
                #Indicate samples whose attribute (name is xi_index) has 
                #a value of xi and class label has a value of c
                X_c_xi = X.ix[c_xi, :]
                #.ix can not only select a cell in a DataFrame via its index 
                #numbers, but also select via its index and column names, as 
                #well as boolean index.
                pc_xi = float((X_c_xi.shape[0] + 1))/pc_xi_denominator
                #This is p(c, xi)
                
                for j in range(self.n):
                    if self.is_continuous[j]:
                        if X_c_xi.shape[0] <= 1:
                            #When the number of samples with specific 
                            #parent attribute value and class label value 
                            #is less than 1, don't consider the parent attribute 
                            #value any more and use the total samples with 
                            #specific class label value to calculate the mean 
                            #and variance of the continuous attribute
                            p_list.append([np.mean(X.ix[y == c, j]), 
                                           np.var(X.ix[y == c, j])])
                        else:
                            #When the number of samples with both specific 
                            #parent attribute value and class label value 
                            #is greater than 1, directly use these samples to 
                            #calculate the mean and variance of the continuous 
                            #attribute
                            p_list.append([np.mean(X_c_xi.ix[:,j]), 
                                           np.var(X_c_xi.ix[:,j])])
                    else:
                         keyval = self.is_continuous.index[j]
                         condi_proba_of_xj = (pd.value_counts(X_c_xi.ix[:,j])[
                                 self.unique_values[keyval]].fillna(0) + 1)/(
    X_c_xi.shape[0] + self.unique_values[keyval].size)
                         #pd.value_counts(values) Compute a histogram of the 
                         #                        counts of non-null values
                         #Returns: value_counts Series
                         p_list.append(np.log(condi_proba_of_xj))
                p_list.append(np.log(pc_xi))
                #Place p(c, xi) to the end of the list
                
                p.ix[xi, c] = p_list
                
        return p
                         
    def fit(self, X):
        """
        Train AODE Bayes classifier with training set X.
        @X
            DataFrame of the training dataset. Each row is a sample, while 
            each column is an attribute, except the last column, which is the 
            class labels of the samples
        """
        y = X.ix[:,(int(X.shape[1]) - 1)]
        X = X.ix[:,0:(int(X.shape[1]) - 1)]
        self.m, self.n = X.shape
        self.unique_y = np.unique(y)
        self.n_class = self.unique_y.size
        
        is_continuous = X.apply(lambda x: (type_of_target(list(x)) == 
                                           'continuous'))
        #type_of_target(y) Determine the type of data indicated by target `y`
        #                  One of: 'continuous', 'binary', 'multiclass', 
        #                          'multiclass-mulitoutpout', 
        #                          'multilabel-indicator', and 'unknown'
        #If want to judge if the values are continuous or not, the values cannot 
        #be included in a Series, otherwise, even if the values are continuous, 
        #will still get 'unknown'. To judge continuous values, change the Series 
        #to a list
        self.is_continuous = is_continuous
        #self.is_discrete = ~is_continuous
        #del self.is_discrete
        
        unique_values = {}
        #A dictionary recording the levels of each discrete attribute. Each 
        #key corresponds to one discrete attribute and its value is the 
        #levels of that discrete attribute
        for i in is_continuous[~is_continuous].index:
            attrname = i
            unique_values[attrname] = np.unique(X.ix[:,attrname])
        #.ix can not only select a cell in a DataFrame via its index numbers, 
        #but also select via its index and column names, as well as boolean 
        #index.
        self.unique_values = unique_values
        
        #Get the attributes qualified to become parent attributes, should be
        #discrete attributes, and the sample size for ALL of its levels should 
        #be greater than the cutoff
        parent_attribute_index = self._get_parent_attribute(X)
        
        total_p = {}
        #A dictionary recording the probability of P(class label = c, 
        #parent attribute = xi) and P(attribute = xj | class label = c, 
        #parent attribute = xi). Each key corresponds to one parent attribute 
        #and its value is a DataFrame recording these probabilities using 
        #different class label levels as columns and different parent attribute 
        #levels as indexes, and each cell is a list containing these 
        #probablilities on various xj when xi and c are fixed.
        for i in parent_attribute_index:
            p = self._spode_fit(X, y, i)
            total_p[i] = p
        self.total_p = total_p
        
        return self
        #A method can directly return self
        
    def _spode_predict(self, X, p, xi_index):
        """
        @X: 
            DataFrame of test dataset. Each row is a sample, while each column 
            is an attribute.
        @xi_index: 
            ONE parent attribute name.
        @p: 
            A DataFrame recording the probability of P(class label = c, 
            parent attribute = xi) and P(attribute = xj | class label = c, 
            parent attribute = xi), for a specific parent attribute xi. 
            It uses different class label levels as columns and different 
            levels for this specific parent attribute as indexes, and each 
            cell is a list containing these probablilities on various xj when 
            xi and c are fixed.
        """
        sharedattrs = list(set(X.columns) & set(self.total_p.keys()))
        assert len(sharedattrs) > 0
        xi = X.ix[:,xi_index]
        result = pd.DataFrame(np.zeros((X.shape[0], p.shape[1])), 
                              columns = self.unique_y)
        result.index = X.index
        for value in p.index:
            xi_value = xi == value
            X_split = X.ix[xi_value,:]
            if X_split.shape[0] < 1:
                continue
            for c in p.columns:
                p_list = p.ix[value, c]
                for j in range(len(sharedattrs)):
                    sharedattr = sharedattrs[j]
                    sharedattr_idx = self.is_continuous.index.get_loc(sharedattr)
                    if self.is_continuous[sharedattr]:
                        mean_, var_ = p_list[sharedattr_idx]
                        result.ix[xi_vlaue, c] += (
                                -np.log(np.sqrt(2*np.pi)*var_) - (X_split[:, j] - 
                                       mean_)**2/(2*var_**2))
                        
                    else:
                        result.ix[xi_value, c] += p_list[sharedattr_idx][
                                X_split.ix[:,sharedattr]].values
                        #If want to generate a DataFrame or Series with replicated 
                        #row indexes, from a DataFrame or Series with unreplicated 
                        #row indexes, can use a Series containing the replicated 
                        #row indexes to select the rows from the original DataFrame 
                        #or Series, and the replicated rows will be selected 
                        #replicatedly. However, it should be a Series that 
                        #contains the replicated row indexes, it cannot be 
                        #a list.
                        #Use .values to convert the Series to an array
        return result  
    
    def predict(self, X):
        """
        Predict the class labels of samples in test dataset X
        @X: 
            DataFrame of the test dataset. Each row is a sample, while 
            each column is an attribute.
        """
        if self.total_p == None:
            raise Exception('You have to fit first before prediction.')
            
        result = pd.DataFrame(np.zeros((X.shape[0], self.unique_y.shape[0])), 
                              columns = self.unique_y)
        result.index = X.index
        
        for i in self.total_p.keys():
            result += self._spode_predict(X, self.total_p[i], i)
        predictions = self.unique_y[np.argmax(result.values, axis = 1)]
        #np.argmax(a, axis = None) Returns the indices of the maxium values along 
        #                          an axis
        result['label'] = predictions
        result = result.ix[:,['good', 'bad', 'label']]
        result.columns = ['p_good', 'p_bad', 'label']
        
        return result
    
#%%
if __name__ == '__main__':
    aode = AODE(0)
    aode.fit(watermelon)
    predictres = aode.predict(test1)

#%%
%reset
