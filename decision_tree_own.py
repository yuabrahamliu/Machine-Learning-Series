# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:31:59 2018

@author: Yu Liu
"""

#%%
import pandas as pd
from pandas import Series, DataFrame
import re
from math import log

def NodeLabel(label_arr):
    """
    Calculate the appeared label and its counts
    
    @param label_arr: Series for labels
    @return label_count: dict, the appeared label and its counts
    """
    label_count = {}
    for label in label_arr:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count

def ValueCount(data_arr):
    """
    Calculate the appeared value for a categoric attribute and its counts
    
    @param data_arr: Series for an attribute
    @return value_count: dict ,the appeared attribute value and its counts
    """
    value_count = {}
    for label in data_arr:
        if label in value_count:
            value_count[label] += 1
        else:
            value_count[label] = 1
    return value_count

def InfoEnt(label_arr):
    """
    Ent = -sigma(p*log2p)
    
    @param label_arr: Series for labels
    @return ent: information entropy
    """
    ent = 0
    n = len(label_arr)
    label_count = NodeLabel(label_arr)
    
    for key in label_count:
        ent -= (float(label_count[key])/n)*log(float(label_count[key])/n, 2)
    
    return ent

def InfoGain(df, index):
    """
    Gain(D,a) = Ent(D) - sigma((|Dv|/|D|)*Ent(Dv))
    
    @param df: DataFrame of the data set, including att columns and label column.
               Label column is the last column
    @param index: string, the target att name
    @return info_gain
    @return div_value: for discrete value, it's 0
                       for continuous value, it's the division value between 
                       2 samples
    """
    #Calcualte the info ent before dividing
    info_gain = InfoEnt(df.ix[:,-1])
    div_value = 0
    n = len(df[index]) # df[index] is the target att value of the samples
    
    if df[index].dtype == ('float64', 'int64'):
        sub_info_ent = {}
        
        df = df.sort_index(by = index, ascending = 1)
        df = df.reset_index(drop = True)
        
        data_arr = df[index] #Get the att value after sorting
        label_arr = df.ix[:,-1] #Get the label value after sorting
        
        for i in range(n-1):
            div = (data_arr[i] + data_arr[i+1])/2.0
            #Calculate the info ent after dividing
            sub_info_ent[div] = (i+1)*InfoEnt(label_arr[0:i+1])/n + \
            (n -i -1)*InfoEnt(label_arr[i+1:])/n
        div_value, sub_info_ent_min = min(sub_info_ent.items(),
                                          key = lambda x: x[1])
        info_gain -= sub_info_ent_min
        
    else:
        data_arr = df[index]
        label_arr = df.ix[:,-1]
        value_count = ValueCount(data_arr)
        
        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            info_gain -= value_count[key]*InfoEnt(key_label_arr)/n
            
    return info_gain, div_value

def OptAttr(df):
    """
    Find the optimal attribute of current data set
    
    @param df: the DataFrame of the data set, the last column should be the 
               label column and the others are the attribute columns
    @return opt_attr: the optimal attribute
    @return div_value: for discrete variable, it's 0
                       for continuous variable, it's the division value between
                       2 samples
    """
    info_gain = 0
    for attr_id in df.columns[:-1]:
        info_gain_tmp, div_value_tmp = InfoGain(df, attr_id)
        if info_gain_tmp > info_gain:
            info_gain = info_gain_tmp
            opt_attr = attr_id
            div_value = div_value_tmp
    return opt_attr, div_value

def Gini(label_arr):
    """
    Gini = 1 - sigma(p^2)
    
    @param label_arr: Series for labels
    @return gini
    """
    gini = 1
    n = len(label_arr)
    label_count = NodeLabel(label_arr)
    for key in label_count:
        gini -= (float(label_count[key])/n)*(float(label_count[key])/n)
    return gini

def GiniIndex(df, index):
    """
    Gini_index(D, a) = sigma((|Dv|/|D|)*Gini(Dv))
    
    @param df: DataFrame of the data set, the last column should be the label
               column, others should be attribute columns
    @param index: string, the target att name
    @return gini_index: the Gini_index of current att
    @return div_value: for discrete value, it's 0
                       for continuous value, it's the division value between 
                       2 samples
    """
    gini_index = 0
    div_value = 0
    n = len(df[index])
    
    if df[index].dtype == ('float64', 'int64'):
        sub_gini = {}
        
        df = df.sort_index(by = [index], ascending = 1)
        df = df.reset_index(drop = True)
        
        data_arr = df[index]
        label_arr = df.ix[:,-1]
        for i in range(n-1):
            div = (data_arr[i] + data_arr[i+1])/2
            sub_gini[div] = ((i+1)*Gini(label_arr[0:i+1])/n) + \
                            ((n-i-1)*Gini(label_arr[i+1:])/n)
        div_value, gini_index = min(sub_gini.items(), 
                                    key = lambda x: x[1])
    else:
        data_arr = df[index]
        label_arr = df.ix[:,-1]
        value_count = ValueCount(data_arr)
        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            gini_index += value_count[key]*Gini(key_label_arr)/n
    return gini_index, div_value

def OptAttr_Gini(df):
    """
    Find the optimal attribute of current data set based on Gini index
    
    @param df: DataFrame of the data set, the last column is the label column, 
               while the others are attribute columns
    @return opt_attr: the optimal attribute
    @return div_value: for discrete variable, it's 0
                       for continuous variable, it's the division value between 
                       2 samples
    """
    gini_index = float('Inf')
    for attr_id in df.columns[:-1]:
        gini_index_tmp, div_value_tmp = GiniIndex(df, attr_id)
        if gini_index_tmp < gini_index:
            gini_index = gini_index_tmp
            opt_attr = attr_id
            div_value = div_value_tmp
    return opt_attr, div_value

class Node(object):
    """
    definition of decision node class
    
    attr: attribute as parent for a new branch
    attr_down:  dict {key, value}
                key: categoric: categoric attr_value
                     continuous: '<= div_value' for small part
                                '> div_value' for big part
                value: children (Node class)
    label: the majority of current sample labels
    """
    def __init__(self, attr_init = None, attr_down_init = {}, 
                 label_init = None):
        self.attr = attr_init
        self.attr_down = attr_down_init
        self.label = label_init

def TreeGenerate(df, method="CART"):
    """
    Generate decision tree using recursion
    
    @param df: the DataFrame of the data set. The last column is label 
               column, while the others are attribute columns
    @param method: the method used to determine the optimal attribute to 
                   branch the tree, "ID3" or "CART"
    @return root: Node instance, the root node of the decision tree
    """
    new_node = Node(attr_init = None, attr_down_init = {}, 
                    label_init = None)
    label_arr = df.ix[:,-1]
    label_count = NodeLabel(label_arr)
    if label_count:
        #Set the label attribute of the new_node instance
        new_node.label = max(label_count.items(), key = lambda x:x[1])[0]
        #End if there is only 1 label class in current node data or 
        #attribute array is empty
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node
        #Set the attr attribute of the new_node instance
        #Get the optimal attribute for new braching
        if method == "CART":
            new_node.attr, div_value = OptAttr_Gini(df)
        else:
            new_node.attr, div_value = OptAttr(df)
        
        #recursion
        if div_value == 0:
            value_count = ValueCount(df[new_node.attr])
            for value in value_count:
                df_v = df[df[new_node.attr].isin([value])]
                #Get subset Dv with value av on attribute a from the original
                #data D
                df_v = df_v.drop(new_node.attr, 1)
                new_node.attr_down[value] = TreeGenerate(df_v, method = method)
                #Set the attr_down attribute of the new_node instance
                #Recursion, the attribute attr_down is another TreeGenerate() 
                #result, which is a Node instance
        else:
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            df_v_l = df[df[new_node.attr] <= div_value]
            df_v_r = df[df[new_node.attr] > div_value]
            
            new_node.attr_down[value_l] = TreeGenerate(df_v_l, method = method)
            new_node.attr_down[value_r] = TreeGenerate(df_v_r, method = method)
    return new_node

def Predict(root, df_sample):
    """
    Predict the label of the sample based on the trained decision tree
    
    @param root: Node, root node of the trained decision tree
    @param df_sample: DataFrame, the attribute vector of a sample, 
                      cannot be Series
    @return: the label of the test sample
    """
    while root.attr != None:
        if df_sample[root.attr].dtype == ('float64', 'int64'):
            for key in list(root.attr_down):
            #list() return the list only containing dict keys
                num = re.findall(r"\d+\.?\d*", key)
                div_value = float(num[0])
                #The matched value returned by re.findall is in a list, 
                #even if only one element, use [0] to extract the element
                break
            if df_sample[root.attr].values[0] <= div_value:
                #The values method of DataFrame returns the value array
                key = "<=%.3f" % div_value
                root = root.attr_down[key]
            else:
                key = ">%.3f" % div_value
                root = root.attr_down[key]
        else:
            key = df_sample[root.attr].values[0]
            if key in root.attr_down:
                root = root.attr_down[key]
            else:
                break
                #Use root = root.attr_down[key] combined with while loop to 
                #achieve the continuous downward search process until meet 
                #the final level of node matching the quried sample
    return root.label

def PredictAccuracy(root, df_test):
    """
    Calculate the accuracy of prediction on test data set
    
    @param root: Node, root node of the trained decision tree
    @param df_test: DataFrame, the test data set, the last column should be 
                    the sample label column, while others should be attribute 
                    columns
    @return: the accuracy of prediction on test data set
    """
    if len(df_test) == 0:
        return 0
    pred_true = 0
    for i in df_test.index:
        label = Predict(root, df_test.ix[:,:-1][df_test.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1
    return float(pred_true)/len(df_test)


def PrePruning(df_train, df_test, method = "CART"):
    """
    Generate a decision tree using prepruning method
    
    @param df_train: DataFrame of the trainig set, last column is the label 
                     column, others are attribute columns
    @param df_test: DataFrame of the testing set, last column in the label 
                    column, others are attribute columns
    @param method: the method used to determine the optimal attribute to 
                   brach the tree, "ID3" or "CART"
    @return root: Node, root of the decision tree generated by prepruning
    """
    new_node = Node(attr_init=None, attr_down_init={}, label_init=None)
    label_arr = df_train[df_train.columns[-1]]
    label_count = NodeLabel(label_arr)
    if label_count:
        #Set the label attribute of the new_node instance
        new_node.label = max(label_count.items(), key = lambda x: x[1])[0]
        #End if there is only 1 label class in current node data
        #or attribute array is empty
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node
        
        #Calculate the test accuracy up to current node
        a0 = PredictAccuracy(new_node, df_test)
        
        #Set the attr attribute of the new_node instance
        #Get the optimal attribute for a new branch
        if method == "CART":
            new_node.attr, div_value = OptAttr_Gini(df_train)
        else:
            new_node.attr, div_value = OptAttr(df_train)
        
        if div_value == 0:
            value_count = ValueCount(df_train[new_node.attr])
            for value in value_count:
                df_v = df_train[df_train[new_node.attr].isin([value])]
                #for isin, value must be put in a [], even if it is only one
                #element
                df_v = df_v.drop(new_node.attr, 1)
                #?df_v.drop(labels, axis = 0) 
                #return new object with labels in requested axis removed.
                #labels: single label or list-like
                #axis: int or axis name
                new_node_child = Node(attr_init=None, attr_down_init={}, 
                                      label_init=None)
                #Determine the label attribute of new_node_child instance
                label_arr_child = df_v[df_v.columns[-1]]
                label_count_child = NodeLabel(label_arr_child)
                new_node_child.label = max(label_count_child.items(), 
                                           key = lambda x: x[1])[0]
                #Set the attr_down attribute of the new_node instance
                new_node.attr_down[value] = new_node_child
                
            a1 = PredictAccuracy(new_node, df_test)
            #To estimate the accuracy after dividing using the first node, 
            #only the label attribute of its child node is needed. Becaue
            #to a node, the attr and attr_down attributes are the attributes
            #describing the variable used to further divding it, namely, they 
            #describe the variable representing the node of the next order, 
            #only the label attribute is used to describe the label status of 
            #the current node! So only the label attribute of its child node
            #is describing the label status of the child node, while the 
            #attr and attr_down attributes representing the variable used to 
            #generate the grandchild nodes
            if a1 > a0:
                for value in value_count:
                    df_v = df_train[df_train[new_node.attr].isin([value])]
                    df_v = df_v.drop(new_node.attr, 1)
                    new_node.attr_down[value] = PrePruning(df_train = df_v, 
                                      df_test = df_test, method = method)
            else:
                new_node.attr = None
                new_node.attr_down = {}
                #These 2 attributes are None or {}, representing the node will 
                #not be further divided. They represent the variable used to
                #generate the node of the next order
                #But label attribute is used to describe the label status of 
                #the current node
        else:
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            df_v_l = df_train[df_train[new_node.attr]] <= div_value
            df_v_r = df_train[df_train[new_node.attr]] > div_value
            #For child node
            new_node_l = Node(attr_init=None, attr_down_init={}, 
                              label_init=None)
            new_node_r = Node(attr_init=None, attr_down_init={}, 
                              label_init=None)
            #Determine the label attribute of the label_count_l and 
            #label_count_r instances
            label_count_l = NodeLabel(df_v_l[df_v_l.columns[-1]])
            label_count_r = NodeLabel(df_v_r[df_v_r.columns[-1]])
            new_node_l.label = max(label_count_l.items(), 
                                   key = lambda x: x[1])[0]
            new_node_r.label = max(label_count_r.items(), 
                                   key = lambda x: x[1])[0]
            #Set the attr_down attribute of the new_node instance
            new_node.attr_down[value_l] = new_node_l
            new_node.attr_down[value_r] = new_node_r
            
            a1 = PredictAccuracy(new_node, df_test)
            if a1 > a0:
                new_node.attr_down[value_l] = PrePruning(df_train=df_v_l, 
                                  df_test=df_test, method=method)
                new_node.attr_down[value_r] = PrePruning(df_train=df_v_r, 
                                  df_test=df_test, method=method)
            else:
                new_node.attr = None
                new_node.attr_down = {}
    return new_node

def PostPruning(root, df_test):
    """
    Generate a decision tree using postpruning method
    
    @param root: Node, root of a decision tree
    @param df_test: DataFrame of the testing set, the last column should be 
                    label column, while others are attribute columns
    @return accuracy score through traversing the tree: 
    """
    if root.attr == None:
        return PredictAccuracy(root, df_test)
    
    a1 = 0
    #Use root.attr to divide the current node and calculate the accuracy 
    #after divding
    value_count = ValueCount(df_test[root.attr])
    for value in list(value_count):
        df_test_v = df_test[df_test[root.attr].isin([value])]
        if value in root.attr_down:
        #This if branch is necessary because for some attributes, although
        #the whole data set has all of its values, for a subset, 
        #it maybe not contain all the values of the attribute
            a1_v = PostPruning(root.attr_down[value], df_test_v)
            #Recursion, the process will continue until a1_v can be calculated 
            #by PredictAccuracy() due to value not in root.attr_down or 
            #root.attr == None
            #Because here the root of the orignial tree has been cut, 
            #the first node to divide the data has been lost, correspondingly, 
            #the test data shoud be cut to fit the dowstream tree branch
        else:
            a1_v = PredictAccuracy(root, df_test_v)
            #Because the current tree without the first node doesn't have 
            #a node can accommodate the df_test_v subset (value is not in 
            #root.attr_down), while all the values in the whole data set 
            #can be accommodated by the complete tree. It means the value is 
            #accommodated by the cut node. So this subset should 
            #fit the complete decision tree representing by root
        if a1_v == -1:
            #-1 means no pruning back from this child
            return -1
        else:
            a1 += a1_v*len(df_test_v.index)/len(df_test.index)
            #The final a1 is the weighted sum of the accuracy of different 
            #subsets with different values of an attribute
    
    node = Node(attr_init=None, attr_down_init={}, label_init=root.label)
    #The label attribute of a node reflects its label status before divding, 
    #so use label attribute can calculate the accuracy before dividing
    a0 = PredictAccuracy(node, df_test)
    
    if a0 > a1:
        root.attr = None
        root.attr_down = {}
        return a0
    else:
        return -1
#%%
if __name__ == "__main__":
    
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
    watermelon.columns = ['Idx', 'color', 'root', 'knocks', 'texture', 
                           'navel', 'touch', 'density', 'sugar_ratio', 'label']
    watermelon = watermelon.set_index('Idx', drop = True)
    
    del Idx, color, root, knocks, texture, navel, touch, density, \
    sugar_ratio, label

    test_index = [4, 5, 8, 9, 11, 12, 13]
    test_data = watermelon.ix[test_index]
    train_data = watermelon[~watermelon.index.isin(test_index)]
    test_data = test_data.reset_index(drop = True)
    train_data = train_data.reset_index(drop = True)
    
    del test_index, watermelon
    
    df_test = test_data
    df_train = train_data
    
    del test_data, train_data
    
    root = TreeGenerate(df_train)
    a = PredictAccuracy(root, df_test)
    root_pre = PrePruning(df_train, df_test, method = "CART")
    a_pre = PredictAccuracy(root_pre, df_test)
    a_post = PostPruning(root, df_test)
