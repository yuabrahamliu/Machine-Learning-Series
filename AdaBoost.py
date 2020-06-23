# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:36:32 2020

@author: abrah
"""

#%%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from matplotlib import pyplot as plt

Idx = Series(range(1, 18))

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

watermelon = pd.concat([Idx, density, sugar_ratio, label], axis = 1)
#pd.concat(objs, axis = 0) Concatenate pandas objects along a particular axis 
#                          with optional set logic along the other axes.
#objs: a sequence or mapping of Series, DataFrame, or Panel objects
#axis: The axis to concatenate along

watermelon.columns = ['Idx', 'density', 'sugar_ratio', 'label']
#Use watermelon.columnS to rename columns

watermelon = watermelon.set_index('Idx', drop = True)
#Set the DataFrame index (row labels) using one or more existing columns. By 
#default yeilds a new object.

watermelon['label'] = watermelon['label'].map({0:'bad', 1:'good'})
#map(function, sequence) Return a list of the results of applying the function 
#                        to the ITEMs of the argument sequence

del Idx, density, sugar_ratio, label

test1 = DataFrame([[1, 0.697, 0.46, 'notknown']], 
                  columns = watermelon.columns.insert(0, 'Idx'))
#Use watermelon.columns.insert(loc, item) to make new Index object with new 
#item inserted at location loc.

test1 = test1.set_index('Idx', drop = True)
#Set the DataFrame index (row labels) using one or more existing columns. By 
#default yields a new object.

#%%
class Node(object):
    def __init__(self):
        self.feature_index = None #feature_index is the column index of feature 
                                  #selected as the optimal classification feature
        self.split_point = None   #split_point is the optimal classification 
                                  #value to discretize the continuous feature
        self.deep = None          #deep is the depth of a decision tree and 
                                  #it starts from 0, i.e., for a decision stump 
                                  #with a depth of 1, its deep value is 0.
        self.left_tree = None     #left_tree is the branch consisting of samples 
                                  #with an optimal classification feature value 
                                  #less than split_point value
        self.right_tree = None    #right_tree is the branch consisting of samples 
                                  #with an optimal classification feature value 
                                  #greater than split_point value 
        self.leaf_class = None    #leaf_class is the predicted label of samples 
                                  #in the leaf node
        
    #During the process of writing a class, if want to test its methods being 
    #written, an annoyance is the error "NameError: name 'self' is not defined". 
    #To get a self object and use it conviently during method writing and testing, 
    #can write the __init__ method of the class first, and generate an object 
    #named 'self' using the temporary class containing the __init__ method only, 
    #like self = Node() here, and then this self can be used to test other 
    #methods need to be written.
    
    
def gini(y, D):
    """
    Calculate the weighted Gini VALUE in a specific dataset
    @y: Series, sample labels of the specific dataset
    @D: Series, sample weights of the specific dataset
    """
    unique_class = np.unique(y)
    total_weight = np.sum(D)
    
    gini = 1
    for c in unique_class:
        gini -= (np.sum(D[list(y == c)])/total_weight)**2
        #If both D and y are Series, they should have the same index, to 
        #select elements in D using D[y == c], where y == c is also a Series.
        #Otherwise, use D[list(y == c)] to change y == c to a list then use this 
        #list to select elements in D
        
    return gini
    
def calcMinGiniIndex(a, y, D):
    """
    Calculate the weighted Gini INDEX in a sepcific dataset after it is 
    splitted by a continuous attribute
    @a: Series, continuous attribute values of the specific dataset
    @y: Series, sample labels of the specific dataset
    @D: Series, sample weights of the specific dataset
    """
    feature = a.sort_values(axis = 0)
    #a.sort_values(axis = 0, ascending = True) Sort by the values along 
    #                                          either axis
    feature.index = range(1, feature.shape[0]+1)
    #The original index will also be sorted along with the value, so after 
    #sorting, it will be out of order and need to be redefined as an order 
    #from 1 to the end
    
    #Use feature.index to rename indeces
    
    total_weight = np.sum(D)
    split_points = [(feature[i] + feature[i + 1])/2 for i in \
                    list(feature.index[range(0, feature.shape[0]-1)])]
    #Note, after the line continuation character "\", there should be NO BLANK
    
    min_gini = float('inf')
    min_gini_point = None
    
    for i in split_points: 
        yv1 = y[list(a <= i)]
        yv2 = y[list(a > i)]
        
        Dv1 = D[list(a <= i)]
        Dv2 = D[list(a > i)]
        #If both D and a are Series, they should have the same index, to 
        #select elements in D using D[a <= i], where a <= i is also a Series. 
        #Otherwise, use D[list(a <= i)] to change a <= i to a list then use this 
        #list to select elements in D
        
        gini_tmp = (np.sum(Dv1)*gini(yv1, Dv1) + np.sum(Dv2)*gini(yv2, Dv2))/total_weight
        
        if gini_tmp < min_gini:
            min_gini = gini_tmp
            min_gini_point = i
            
    return min_gini, min_gini_point

def chooseFeatureToSplit(X, D):
    """
    Choose the optimal classification feature (continuous) and the optimal 
    classification value to discretize this continous feature
    @X: DataFrame, each row is a sample and each column is a feature, except the 
        last column, which is the sample labels. The feature colunms should be 
        continuous values
    @D: Series, sample weights
    """
    y = X.ix[:,X.shape[1]-1]
    X = X.ix[:,0:X.shape[1]-1]
    
    ginilist = []
    split_pointlist = []
    for i in range(0, X.shape[1]): 
        giniindex, split_point = calcMinGiniIndex(X.ix[:, i], y, D)
        ginilist.append(giniindex)
        split_pointlist.append(split_point)
        
    ginilist = Series(ginilist)
    split_pointlist = Series(split_pointlist)
    
    minigini = min(ginilist)
    minsplit_point = list(split_pointlist[list(ginilist == minigini)])[0]
    #If both split_pointlist and ginilist are Series, they should have the same 
    #index, to select elements in split_pointlist using 
    #split_pointlist[ginilist == minigini], where ginilist == minigini is also 
    #a Series. Otherwise, use split_pointlist[list(ginilist == minigini)] to 
    #change ginilist == minigini to a list then use this list to select elements 
    #in split_pointlist
    colindex = list(Series(range(0, X.shape[1]))[list(ginilist == minigini)])[0]
    colname = X.columns[colindex]
    
    return colindex, minsplit_point

def createSingleTree(X, D, deep = 0, deeplim = 0):
    """
    Generate a C4.5 decision tree (base learner of the Bagging model) 
    using Gini index to find the optimal classification feature and Gini index 
    is calculated based on sample weights
    @X: DataFrame, each row is a sample and each column is a feature, except the 
        last column, which is the sample labels. The feature columns should be 
        continous values
    @D: Series, sample weights
    @deep: the depth of a decision tree and it starts from 0, i.e., for a 
           decision stump with a depth of 1, its deep value is 0.
    @deeplim: the maximum deep value of a decision tree allowed
    """
    node = Node()
    node.deep = deep
    
    Xunique = X.drop_duplicates()
    #drop_duplicates Return DataFrame with duplicate rows removed, optionally 
    #only considering certain columns
    
    if(deep > deeplim) | (X.shape[0] <= 2) | (Xunique.shape[0] <= 1):
        #When the deep value of the decision tree has exceeded the deeplim value, 
        #or when the sample size in the current node is not greater than 2, 
        #or as to a bootstrapped subset, all the current samples are from the 
        #same one sample and have same attribute values, 
        #attribute them as leaf node
        
        weights = []
        unique_class = np.unique(X.ix[:,X.shape[1] - 1])
        for c in unique_class:
            weight = np.sum(D[list(X.ix[:,X.shape[1] - 1] == c)])
            weights.append(weight)
        weights = Series(weights)
        labels = Series(unique_class)
        weights = pd.concat([labels, weights], axis = 1)
        #pd.concat(objs, axis = 0) Concatenate pandas objects along a particular 
        #                          axis with optional set logic along the other 
        #                          axes.
        #objs: a sequence or mapping of Series, DataFrame, or Panel objects
        #axis: The axis to concatenate along
        
        weights.columns = ['labels', 'weights']
        #Use weights.columnS to rename columns
        weights.index = range(1, weights.shape[0] + 1)
        #Use weights.index to rename indeces
        node.leaf_class = list(weights['labels'][list(weights['weights'] == \
                                       max(weights['weights']))])[0]
        #Note, after the line continuation character "\", there should be NO BLANK
   
        return node
    
    feature_index, split_point = chooseFeatureToSplit(X = X, D = D)
    
    node.feature_index = feature_index
    node.split_point = split_point
    
    left = X.ix[:, feature_index] <= split_point
    right = X.ix[:, feature_index] > split_point
    
    node.left_tree = createSingleTree(X = X.ix[list(left),:], 
                                  D = D[list(left)], deep = deep + 1, 
                                  deeplim = deeplim)
    
    node.right_tree = createSingleTree(X = X.ix[list(right),:], 
                                   D = D[list(right)], deep = deep + 1, 
                                   deeplim = deeplim)

    return node

    
def predictSingle(tree, x):
    """
    Predict the label of a single sample according to the base learner tree
    @tree: C4.5 decision tree trained from the training dataset
    @x: Series, each row is a feature, the last row can be sample label or not
    """
    x = x.to_frame()
    #x.to_frame() Convert Series to DataFrame
    #The result DataFrame has 1 column and the same number of rows as the 
    #original Series
    
    x = x.T
    x.index = [1]
    #Use x.index to rename indeces
    
    if tree.leaf_class is not None:
    #Here tree is a C4.5 decision tree object, if this decision tree has a leaf 
    #node on its first layer, then tree.leaf_class will not be None, but if 
    #the leaf node is not on first layer, but sencond, third, or fourth layer, 
    #the leaf_class value will not be saved in tree.leaf_class, but in 
    #tree.left_tree.leaf_class, tree.left_tree.left_tree.leaf_class, 
    #or tree.right_tree.left_tree.right_tree.leaf_class, and so on, while the 
    #value of tree.leaf_class is None.
        return tree.leaf_class
    
    if x.ix[:,tree.feature_index][1] > tree.split_point:
        x = x.ix[1,:]
        #x Should be a Series as defined by the function, so convert it from 
        #DataFrame to Series first
        return predictSingle(tree = tree.right_tree, x = x)
        #Each time of recursion, change the value of parameter tree of the 
        #original function to a deeper layer of the C4.5 tree object, such as 
        #tree = tree.right_tree here (from the original tree = tree), so that 
        #each time of recursion, the search can go further to a deeper layer 
        #of the C4.5 tree, and finally, the whole tree with all the layers 
        #can be searched.
    else:
        x = x.ix[1,:]
        return predictSingle(tree = tree.left_tree, x = x)

def predictBase(tree, X):
    """
    Predict the labels of all samples in a specific DataFrame, according to 
    the base learner tree
    @tree: C4.5 decision tree trained from the training dataset
    @X: DataFrame, each row is a sample and each column is a feature, the 
        last column can be the sample labels or still a feature column. 
        The feature colunms should be continuous values
    """
    result = []
    
    for i in range(X.shape[0]):
        singleindex = X.index[i]
        singlelabel = predictSingle(tree = tree, x = X.ix[singleindex,:])
        result.append(singlelabel)
        
    result = Series(result)
    result.index = X.index
    #Use result.index to rename indeces
    
    return result
    
def adaBoostTrain(X, tree_num = 20, treedeeplim = 0):
    """
    Use C4.5 decision tree as base learner to construct an adaBoost model
    @X: DataFrame, each row is a sample and each column is a feature, except the 
        last column, which is the sample labels. The feature colunms should be 
        continuous values
    @tree_num: the largest number of base learners in the adaBoost model
    @treedeeplim: the maximum deep value of a decision tree allowed. It starts 
                  from 0, i.e., for a decision tree with a depth no greater 
                  than 1, its treedeeplim value is 0.
    """
    y = X.ix[:,X.shape[1]-1]
    y = y.map({'good':1, 'bad':-1})
    #map(function, sequence) Return a list of the results of applying the function 
    #                        to the ITEMs of the argument sequence
    D = np.ones(y.shape)/y.shape
    #Initilize sample weights
    D = Series(D)
    D.index = X.index
    
    trees = []
    #All the base learners
    
    a = []
    #Weigths of the BASE LEARNERS
    
    agg_est = np.zeros(y.shape)
    #Use agg_est to save the accumulated predicted values of base learners
    agg_est = Series(agg_est)
    
    for _ in range(tree_num):
    #Here, _ is a dummy variable
        tree = createSingleTree(X = X, D = D, deep = 0, deeplim = treedeeplim)
        hx = predictBase(tree = tree, X = X)
        hx = hx.map({'good':1, 'bad':-1})
        #map(function, sequence) Return a list of the results of applying the 
        #                        function to the ITEMs of the argument sequence
        err_rate = np.sum(D[list(hx != y)])
        
        at = np.log((1 - err_rate)/max(err_rate, 1e-16))/2
        agg_est += list(at*hx)
        #Both agg_est and at*hx are Series, in the case that they have the 
        #same index, it can be writen directly that agg_est += at*hx, otherwise, 
        #write agg_est += list(at*hx), because if agg_est and at*hx have different 
        #indeces, the index number in agg_est but not in at*hx will correspond 
        #to NA value, rather than the value at the same absolute position in 
        #at*hx.Hence, it is important to guarantee that Serieses have the same 
        #index, to avoid mistake.
        trees.append(tree)
        a.append(at)
        
        if (err_rate > 0.5) | (err_rate == 0):
            break
        
        err_index = np.ones(y.shape)
        err_index = Series(err_index)
        err_index[list(hx == y)] = -1
        D = D*list(np.exp(err_index * at))
        #Both D and np.exp(err_index * at) are Series, in the case that they 
        #have the same index, it can be writen directly that 
        #D = D*np.exp(err_index * at), otherwise, write 
        #D = D*list(np.exp(err_index * at)),because if D and 
        #np.exp(err_index * at) have different indeces, the index number not 
        #shared by them will get an NA value. Hence, it is important to 
        #guarantee that Seireses have the same index, to avoid mistake.
        
        #When handle missing values, decision tree model will give each sample 
        #x a weight wx. In AdaBoost, even if there is no missing value in the 
        #samples, and a single decision tree does not need to have sample weights, 
        #to the whole AdaBoost model, to achieve the update of sample weights, 
        #in a single decision tree model, sample weights can be introduced.
        D = D/np.sum(D)
    
    return trees, a, agg_est

def adaBoostPredict(X, trees, a):
    """
    Use the constructed adaBoost model to predict sample labels
    @X: DataFrame, each row is a sample and each column is a feature, the 
        last column can be sample labels, or also a sample feature column. 
        The feature colunms should be continuous values
    @trees: a list containing all the C4.5 decision trees included in the 
            trained adaBoost model
    @a: a list containing all the BASE LEARNER (C4.5 decision tree here) 
        weights of the trained adaBoost model, corresponding to the base 
        learners in the list trees
    """
    agg_est = np.zeros(X.shape[0])
    agg_est = Series(agg_est)
    
    for tree, am in zip(trees,a):
        #zip Return a list of tuples,  where each tuple contains the i-th 
        #    element from each of the argument sequences.
        agg_est += list(am * np.array(predictBase(tree = tree, X = X).\
                                      map({'good':1, 'bad':-1})))
        #For a list like [1, 2, 3], the result of 3*[1, 2, 3] or [1, 2, 3]*3 
        #is [1, 2, 3, 1, 2, 3, 1, 2, 3]. It is the extension of original list, 
        #not a list with the product between 3 and the origianl elements as its 
        #new elements like [3, 6, 9]. If want to get such a result, should use 
        #array, like 3*array([1, 2, 3]) = array([1, 2, 3])*3 = array([3, 6, 9]). 
        #If want to get array([1, 2, 3, 1, 2, 3, 1, 2, 3]), should use 
        #np.tile(array([1, 2, 3]), 3)
        #Hence, here should use am * np.array(predictBase(tree = tree, X = X).\
        #                                     map({'good':1, 'bad':-1}))
        #to get an array, 
        #or direclty use am * predictBase(tree = tree, X = X).\
        #                                 map({'good':1, 'bad':-1})
        #to get a Series (Because predictBase returns a Series as defined)
        #However, cannot use am * list(predictBase(tree = tree, X = X).\
        #                              map({'good':1, 'bad':-1}))
    agg_est = Series(agg_est)
    agg_est.index = X.index
    
    result = np.ones(X.shape[0],)
    result[list(agg_est < 0)] = -1
    result = Series(result)
    result.index = X.index
    result = result.map({1:'good', -1:'bad'})
    
    return result

def pltAdaBoostDecisionBound(X_, trees, a):
    """
    Draw the decision boundary from the adaBoost model with C4.5 decision trees 
    as base learners
    @X_: DataFrame, each row is a sample and each column is a feature, except the 
         last column, which is the sample labels. The feature colunms should be 
         continuous values
    @trees: a list containing all the C4.5 decision trees included in the 
            trained adaBoost model
    @a: a list containing all the BASE LEARNER (C4.5 decision tree here) 
        weights of the trained adaBoost model, corresponding to the base 
        learners in the list trees
    """
    y_ = X_.ix[:,X_.shape[1]-1]
    X_ = X_.ix[:,0:X_.shape[1]-1]
    pos = y_ == 'good'
    neg = y_ == 'bad'
    
    x_tmp = np.linspace(0, 1, 60)
    y_tmp = np.linspace(-0.2, 0.7, 60)
    #numpy.linspace(start, stop, num = 50) Return evenly spaced numbers over a 
    #                                      specified interval.
    #                                      Return num = 50 (or other) evenly 
    #                                      spaced samples, calculated over the 
    #                                      interval[start, stop].
    #                                      The endpoint of the interval can 
    #                                      optionally be excluded.
    
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    #numpy.meshgrid(*xi) Return coordinate matrices from coordinate vectors
    #                    *xi are coordinate vectors and should be array-like.
    #For example, x = np.array([1, 2, 3]), y = np.array([4, 5, 6, 7]), 
    #xv, yv = np.meshgrid(x, y). Then we can get 
    #xv is array([[1, 2, 3], 
    #             [1, 2, 3], 
    #             [1, 2, 3], 
    #             [1, 2, 3]])
    #yv is array([[4, 4, 4], 
    #             [5, 5, 5], 
    #             [6, 6, 6], 
    #             [7, 7, 7]])
    #The column number of xv = the column number of yv = length of x
    #Each row of xv is x
    #The row number of xv = the row number of yv = length of y
    #Each column of yv is y
    #While if merge xv and yv together, can get
    #array([[(1, 7), (2, 7), (3, 7)], 
    #       [(1, 6), (2, 6), (3, 6)], 
    #       [(1, 5), (2, 5), (3, 5)], 
    #       [(1, 4), (2, 4), (3, 4)]])
    #For this merged array, its shape is the same as xv and yv. The x coordinates
    #of the dots are from xv while the y coordinates of the dots are from yv. 
    #For the coordinates of the dots, they will become the crosses of grid lines 
    #parallel to x axis and y axis in the coordinate system.
    
    Z_ = adaBoostPredict(X = DataFrame(np.c_[X_tmp.ravel(), Y_tmp.ravel()]), \
                         trees = trees, a = a).reshape(X_tmp.shape)
    #numpy.ravel Return a contiguous flattened array
    #numpy.c_ Translates slice objects to concatenation along the second axis
    #reshape return an NDARRAY with the values shape
    Z_ = DataFrame(Z_)
    Z_ = Z_.replace(to_replace = ['good', 'bad'], value = [1, -1])
    #DataFrame.replace(to_replace = None, value = None) 
    #    Replace values given in 'to_replace' with 'value'
    #    to_replace: str, regex, list, dict, Series, numeric, or None
    #    value: scalar, dict, list, str, regex, default None
    
    plt.contour(X_tmp, Y_tmp, Z_, [0], color = 'orange', linewidths = 1)
    #plt.contour(X, Y, Z, levels, **kwargs) Draw contour lines.
    #    X, Y: array-like, optional. The coordinates of the values in Z. X and Y 
    #          must both be 2-D with the same shape as Z (e.g. created via 
    #          numpy.meshgrid), or they must both be 1-D such that len(X) == M 
    #          is the number of columns in Z and len(Y) == N is the number of 
    #          rows in Z.
    #    Z: array-like(N, M) The height values over which the contour is drawn.
    #    levels: int or array-like, optional. Determines the number and positions 
    #            of the contour lines/regions. If an int n, use n data intervals; 
    #            i.e. draw n+1 contour lines. The level heights are automatically 
    #            chosen. If array-like, draw contour lines at the specific levels, 
    #            The values must be in increasing order.
    
    plt.scatter(X_.ix[pos, 0], X_.ix[pos, 1], label = '1', color = 'c')
    plt.scatter(X_.ix[neg, 0], X_.ix[neg, 1], label = '0', color = 'lightcoral')
    #plt.scatter(x, y) Make a scatter plot of `x` vs `y`
    #    x, y: array_like, shape(n,)
    plt.legend()
    #plt.legend Place a legend on the axes.
    #This method can automatically detect the elements to be shown in the legend
    #The elements to be added to the legend are automatically determined, when 
    #you do not pass in any extra arguments.
    #In this case, the LABELs (LABEL = '1' here) are taken from the 
    #artist(plt.scatter(X_.ix[pos, 0], X_.ix[pos, 1], LABEL = '1', color = 'c') 
    #here). You can specify them either at artist creation or by calling the 
    #set_label() method on the artist.
    plt.show()
    #plt.show Display a figure. When running in ipython with its pylab mode, 
    #         display all figures and return to the ipython prompt.
    

    
#%%
if __name__ == '__main__':
    trees, a, agg_est = adaBoostTrain(X = watermelon, tree_num = 20, 
                                      treedeeplim = 1)
    pltAdaBoostDecisionBound(X_ = watermelon, trees = trees, a = a)
    
   
#%%
%reset

    
        
        






