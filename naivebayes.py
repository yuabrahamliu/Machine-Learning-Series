# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 13:40:18 2020

@author: abrah
"""

#%%
import pandas as pd
from pandas import Series, DataFrame
import math
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
class LaplacianNB():
    """
    Laplacian naive bayes for binary classification problem.
    """
    def __init__(self):
        """
        """
        #Even if no content need to be written in the __init__ method, the """
        #must be written here, otherwise, there will be no sign indicating the 
        #end of def __init__(self): and the next new method will be deemed as 
        #the content of this __init__ method.
    def count_list(self, l):
        """
        Get unique elements in list and corresponding counts
        """
        unique_dict = {}
        for e in l:
            if e in unique_dict:
                unique_dict[e] += 1
            else:
                unique_dict[e] = 1
        return unique_dict
    
    def discrete_p(self, d, N_class, dwhole):
        """
        Compute Laplacian corrected discrete attribute probability.
        Return a dictionary recording the probability law of ONE attribute 
            in ONE dataset with a specific class label. Each key is a possible 
            level of this ONE attribute in the ONE dataset with the specific 
            class label, while the value of this key is the Laplacian corrected 
            frequency of this attribute level in such a labeled dataset
        @d  
            A dictionary representing the sample counts of ONE attribute in ONE 
            dataset with a specific class label. Each key is a possible level 
            of this ONE attribute in the ONE dataset with the specific class 
            label, while the value of this key is the count of the samples with 
            such an attribute level in such a labeled dataset.
        @N_class
            The number of ALL SAMPLEs with the specific CLASS label, without 
            considering ATTRIBUTEs
        @dwhole
            A dictionary representing the sample counts of ONE attribute in 
            the WHOLE dataset, without considering class label. Each key is a 
            possible level of this ONE attribute in the WHOLE dataset, while 
            the value of this key is the count of the samples with such an 
            attribute level in the WHOLE dataset
        """
        new_d = {}
        for a, n in d.items():
            new_d[a] = float(n + 1)/(N_class + len(dwhole))
            #N_class is the number of ALL SAMPLEs with the specific CLASS label, 
            #without considering ATTRIBUTEs
            #len(dwhole) is the number of LEVELs of the ONE specific attribute 
            #in the WHOLE dataset, without considering class label
            #new_d is a dictionary recording the probability law of ONE attribute 
            #in ONE dataset with a specific class label. Each key is a possible 
            #level of this ONE attribute in the ONE dataset with the specific 
            #class label, while the value of this key is the Laplacian corrected 
            #frequency of this attribute level in such a labeled dataset
        return new_d
    
    def continuous_p(self, x, mu, var):
        """
        Compute probability density of ONE specific continuous value from ONE
        SAMPLE
        @x
            The value of a continuous variable (1 dimension) from ONE SAMPLE
        @mu 
            The expectation of the normal distribution of this continuous variable
        @var
            The variance of the normal distribution of this continuous variable
        """
        p = 1.0/(math.sqrt(2*math.pi)*math.sqrt(var))*math.exp(-(x-mu)**2/(2*var))
        return(p)
    
    def mu_var_of_list(self, l):
        """
        Estimate the expectation and variance of a normal distribution based on 
        actual sample values
        @l
            Each element of the list l represents a specific continuous variable 
            value of one sample, while the whole list l represents that of a 
            dataset including many samples
        """
        mu = sum(l)/float(len(l))
        var = 0
        for i in range(len(l)):
            var += (l[i] - mu)**2
        var = var/float(len(l))
        return mu, var
    
    def train(self, X):
        """
        Train Laplacian naive Bayes classifier with training set X.
        @X
            DataFrame of the training dataset. Each row is a sample, while 
            each column is an attribute, except the last column, which is the 
            class labels of the samples
        """
        N = int(X.shape[0])
        y = X.ix[:,(int(X.shape[1]) - 1)]
        y = list(y)
        
        self.classes = self.count_list(y)
        self.class_num = len(self.classes)
        self.classes_p = {}
        for c, n in self.classes.items():
            self.classes_p[c] = float(n + 1)/(N + self.class_num)
        
        is_continuous = X.apply(lambda x: type_of_target(list(x)) == 'continuous')
        #type_of_target(y) Determine the type of data indicated by target `y`
        #                  One of: 'continuous', 'binary', 'multiclass', 
        #                          'multiclass-mulitoutpout', 
        #                          'multilabel-indicator', and 'unknown'
        #If want to judge if the values are continuous or not, the values cannot 
        #be included in a Series, otherwise, even if the values are continuous, 
        #will still get 'unknown'. To judge continuous values, change the Series 
        #to a list
        continuousset = X.ix[:,is_continuous]
        discreteset = X.ix[:,~is_continuous]
        discreteset = discreteset.ix[:,0:(discreteset.shape[1] - 1)]
        
        self.discrete_attris_with_good_p = {}
        self.discrete_attris_with_bad_p = {}
        for i in range(discreteset.shape[1]):
            attr_name = discreteset.columns[i]
            attr_with_good = []
            attr_with_bad = []
            attr_whole = []
            for j in discreteset.index:
                attr_whole.append(discreteset.ix[j, i])
                if y[j - 1] == 'good':
                    attr_with_good.append(discreteset.ix[j, i])
                else:
                    attr_with_bad.append(discreteset.ix[j, i])
            unique_whole = self.count_list(attr_whole)
            unique_with_good = self.count_list(attr_with_good)
            unique_with_bad = self.count_list(attr_with_bad)
            self.discrete_attris_with_good_p[attr_name] = self.discrete_p(unique_with_good, \
                                                                          self.classes['good'], \
                                                                          unique_whole)
            self.discrete_attris_with_bad_p[attr_name] = self.discrete_p(unique_with_bad, \
                                                                         self.classes['bad'], \
                                                                         unique_whole)
            
        self.good_mus = {}
        self.good_vars = {}
        self.bad_mus = {}
        self.bad_vars = {}
        for i in range(continuousset.shape[1]):
            attr_name = continuousset.columns[i]
            attr_with_good = []
            attr_with_bad = []
            for j in continuousset.index:
                if y[j - 1] == 'good':
                    attr_with_good.append(continuousset.ix[j, i])
                else:
                    attr_with_bad.append(continuousset.ix[j, i])
            good_mu, good_var = self.mu_var_of_list(attr_with_good)
            bad_mu, bad_var = self.mu_var_of_list(attr_with_bad)
            self.good_mus[attr_name] = good_mu
            self.good_vars[attr_name] = good_var
            self.bad_mus[attr_name] = bad_mu
            self.bad_vars[attr_name] = bad_var
            
    def predict(self, x):
        """
        Predict the class label of sample x, using the Laplacian navie Bayes 
        classifier trainned
        @x
            DataFrame containing the test dataset, one row is one sample, 
            while one column is one attribute.
        """
        p_good = self.classes_p['good']
        p_bad = self.classes_p['bad']
        #Actually, classes_p was calculated within the previous method train(), 
        #and if want to use it here, one way is to transfer it via the 
        #parameters of the current method predict(), which is generally used 
        #when writing an ordinary function (not method), while the other way 
        #is specific for the methods of a class, the object needs to be used 
        #(like classes_p here) can directly be transferred to the current 
        #method using self.classes_p, without specially transferring it using 
        #the parameters of current method.
        
        sharedcontinuous = set(x.columns) & set(self.good_mus.keys())
        sharedcontinuous = list(sharedcontinuous)
        shareddiscrete = set(x.columns) & set(self.discrete_attris_with_good_p.keys())
        shareddiscrete = list(shareddiscrete)
        continuousx = x.ix[:,sharedcontinuous]
        discretex = x.ix[:,shareddiscrete]
        
        labels = []
        p_goods = []
        p_bads = []
        for i in continuousx.index:
            for j in range(discretex.shape[1]):
                dis_attr_name = discretex.columns[j]
                p_good *= self.discrete_attris_with_good_p[dis_attr_name][discretex.ix[i,j]]
                p_bad *= self.discrete_attris_with_bad_p[dis_attr_name][discretex.ix[i,j]]
            for j in range(continuousx.shape[1]):
                con_attr_name = continuousx.columns[j]
                p_good *= self.continuous_p(continuousx.ix[i,j], self.good_mus[con_attr_name], \
                                            self.good_vars[con_attr_name])
                p_bad *= self.continuous_p(continuousx.ix[i,j], self.bad_mus[con_attr_name], \
                                           self.bad_vars[con_attr_name])
            if p_good >= p_bad:
                labels.append('good')
            else:
                labels.append('bad')
            p_goods.append(p_good)
            p_bads.append(p_bad)
        res = DataFrame({'idx': range(1, (x.shape[0]+1)), 
                         'p_good': p_goods, 'p_bad': p_bads, 'label': labels})
        res = res[['idx', 'p_good', 'p_bad', 'label']]
        #Use DataFrame[[new column name order]] to change the order of columns
        res = res.set_index('idx', drop = True)
        #Set the DataFrame index (row labels) using one or more existing columns. 
        #By default yields a new object.
        return res
    
#%%
if __name__ == '__main__':
    lnb = LaplacianNB()
    lnb.train(watermelon)
    predictres = lnb.predict(test1)

#%%    
%reset
