# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:09:55 2018

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

def TreeGenerate(df):
    """
    Generate decision tree using recursion
    
    @param df: the DataFrame of the data set. Last column should be label 
               column, while others should be attribute columns
    @return root: Node instance, the root node of the decision tree
    """
    new_node = Node(attr_init=None, attr_down_init={}, label_init=None)
    label_arr = df.ix[:,-1]
    label_count = NodeLabel(label_arr)
    if label_count:
        #Set the label attribute of new_node instance
        new_node.label = max(label_count.items(), 
                             key = lambda x: x[1])[0]
        #End if there is only 1 class in current node data or the attribute
        #array is empty
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node
        
        #Get the optimal attribute for a new branching
        #Set the attr attribute of new_node instance
        new_node.attr, div_value = OptAttr(df)
        
        if div_value == 0:
            value_count = ValueCount(df[new_node.attr])
            for value in value_count:
                df_v = df[df[new_node.attr].isin([value])]
                #Get the subset with a value V on attribute a in set D, 
                #i.e. Dv
                df_v = df_v.drop(new_node.attr, axis = 1)
                new_node.attr_down[value] = TreeGenerate(df_v)
                #Set the attr_down attribute of new_node instance
                #Recursion, the attr_down attribute is another TreeGenerate() 
                #result, which is a Node instance
        else:
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            df_v_l = df[df[new_node.attr] <= div_value]
            df_v_r = df[df[new_node.attr] > div_value]
            new_node.attr_down[value_l] = TreeGenerate(df_v_l)
            new_node.attr_down[value_r] = TreeGenerate(df_v_r)
        
        return new_node

def Predict(root, df_sample):
    """
    Predict the label of the test sample based on the trained decision tree
    
    @param root: Node, root node of the trained decision tree
    @param df_sample: DataFrame, the attribute vector of a test sample, 
                      cannot be Series
    @return: the label of the test sample
    """
    while root.attr != None:
        if df_sample[root.attr].dtype == ('float64', 'int64'):
            for key in list(root.attr_down):
            #list return the list only containing dict keys
                num = re.findall(r"\d+\.?\d*", key)
                div_value = float(num[0])
                #The matched value returned in re.findall is in a list, 
                #even if only one element. Use [0] to extract the value
                break
            if df_sample[root.attr].values[0] <= div_value:
                #The values method of DataFrame return the value array
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
    
    
    
    df = watermelon.ix[2:,]
    df_sample = DataFrame(watermelon.ix[1,]).T
    
    root = TreeGenerate(df)
    Predict(root, df_sample)
    
    
                                                                                    
                    
    
        
    
                    
