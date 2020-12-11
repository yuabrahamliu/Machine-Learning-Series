# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:57:27 2019

@author: abrah
"""

#%%Load and normalize data
import os
os.chdir('D:\\machine_learning')

from sklearn.datasets import load_breast_cancer

data_set = load_breast_cancer()

X = data_set.data
feature_names = data_set.feature_names
y = data_set.target #label
target_names = data_set.target_names

import matplotlib.pyplot as plt

f1 = plt.figure(1)
#Creates a new figure

p1 = plt.scatter(X[y==0, 0], X[y==0, 1], color = 'r', label = target_names[0])
p2 = plt.scatter(X[y==1, 0], X[y==1, 1], color = 'g', label = target_names[1])
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(loc = 'upper right')
plt.grid(True, linewidth = 0.3)
#Turn the axes grids on or off.

plt.show()

#data normalization
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)

#preprocessing.normalize
#Scale input vectors individually to unit norm (vector length)
#norm: 'l1', 'l2', or 'max', optional ('l2' by default)
#The norm to use to normalize each non zero sample (or each non-zero feature 
#if axis is 0).
#axis: 0 or 1, optional (1 by default)
#axis used to normalize the data along. If 1, independently normalize each 
#sample, otherwise (if 0) normalize each feature.

#%%SVM
#model fitting and testing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm
import numpy as np

#Generation of train set and testing set
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, 
                                                    test_size = 0.5, 
                                                    random_state = 0)
#random_state: int or RandomState
#Pseudo-random number generator state used for random sampling
#The data need to be normalized here

#Model fitting, testing, visualization
#Based on linear kernel and rbf kernel
for fig_num, kernel in enumerate(('linear', 'rbf')):
    #enumerate return an enumerate object.
    #The enumerate object yields pairs containing a count 
    #(from start, which defaults to zero) and a value yielded by the iterable 
    #argument.
    #enumerate is useful for obtaining an indexed list:
        #(0, seq[0]), (1, seq[1]), (2, seq[2]), ...
    
    accuracy = []
    c = []
    
    for C in range(1, 1000, 1):
        clf = svm.SVC(C = C, kernel = kernel)
        #C-Support Vector Classification
        #C: Penalty parameter C of the error term (default = 1.0)
        
        #train
        clf.fit(X_train, y_train)
        
        #testing
        y_pred = clf.predict(X_test)
        
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        #Accuray classification score.
        #normalize: bool, optional (default = True)
        #If ``False``, return the number of correctly classified samples.
        #Otherwise, return the fraction of correctly classified samples.
        
        c.append(C)
        
    print('max accuracy of %s kernel SVM: %.3f' % (kernel, max(accuracy)))
    
    #Draw accuracy
    f2 = plt.figure(2)
    plt.plot(c, accuracy)
    plt.xlabel('penalty parameter')
    plt.ylabel('accuracy')
    
    plt.show()
    
#%%BP network
#Construction of data in pybrain's formation
from pybrain.datasets import ClassificationDataSet

ds = ClassificationDataSet(30, 1, nb_classes=2, class_labels=y)
#Specialized data set for classification data. Classes are to be numbered from 
#0 to nb_class - 1.
#Initialize an empty dataset
#`inp` is used to specify the dimensional of the input.
#The number of targets is given implicitly by the training samples, it can 
#also be set explicitly by `nb_classes`
#To give the classes names, supply an iterable of strings as `class_labels`

for i in range(len(y)):
    ds.appendLinked(normalized_X[i], y[i])
    #Add rows to all linked fields at once.
    #The data need to be normalized here
    
ds.calculateStatistics()
#Return a class histogram

#Split of training and testing dataset
tstdata_temp, trndata_temp = ds.splitWithProportion(0.5)
#Produce 2 new datasets, the first one contains the fraction given by 
#`proportion` of the samples
tstdata = ClassificationDataSet(30, 1, nb_classes=2)
#Specialized data set for classification data. Classes are to be numbered from 
#0 to nb_class - 1.
#Initialize an empty dataset
#`inp` is used to specify the dimensional of the input.
#The number of targets is given implicitly by the training samples, it can 
#also be set explicitly by `nb_classes`

for n in range(0, tstdata_temp.getLength()):
    #Return the length of the linked data fields (sample number).
    tstdata.appendLinked(tstdata_temp.getSample(n)[0], 
                         tstdata_temp.getSample(n)[1])
    #appendLinked
    #Add rows to all linked fields at once.
    #getSample
    #Return a sample at `index` or the current sample

trndata = ClassificationDataSet(30, 1, nb_classes=2)
#Specialized data set for classification data. Classes are to be numbered from 
#0 to nb_class - 1.
#Initialize an empty dataset
#`inp` is used to specify the dimensional of the input.
#The number of targets is given implicitly by the training samples, it can 
#also be set explicitly by `nb_classes`

for n in range(0, trndata_temp.getLength()):
    #Return the length of the linked data fields (sample number).
    trndata.appendLinked(trndata_temp.getSample(n)[0], 
                         trndata_temp.getSample(n)[1])
    #appendLinked
    #Add rows to all linked fields at once.
    #getSample
    #Return a sample at `index` or the current sample
    
trndata._convertToOneOfMany()
#Converts the target classes to a 1-of-k representation, retaining the 
#old targets as a field `class`
tstdata._convertToOneOfMany()
#Converts the target classes to a 1-of-k representation, retaining the 
#old targets as a filed `class`

#Build net and training
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

n_hidden = 500
bp_nn = buildNetwork(trndata.indim, n_hidden, trndata.outdim, 
                     outclass = SoftmaxLayer)
#Build arbitrarily deep networks
#`layers` should be a list or tuple of integers, that indicate how many 
#neurons the layers should have.
trainer = BackpropTrainer(bp_nn, 
                          dataset = trndata, 
                          verbose = True, 
                          momentum = 0.5, 
                          learningrate = 0.0001, 
                          batchlearning = True)
#The learning rate gives the ratio of which parameters are changed into the 
#direction of the gradient. The learning rate decreases by `lrdecay`, which 
#is used to multiply the learning rate after each training step.
#The parameters are also adjusted with respect to `momentum`, which is the 
#ratio by which the gradient of the last timestep is used.
#If `batchlearning` is set, the parameters are updated only at the end of 
#each epoch. Default is False.

err_train, err_valid = trainer.trainUntilConvergence(maxEpochs = 1000, 
                                                     validationProportion = 0.25)
#If no dataset is given, the dataset passed during Trainer initialization is 
#used. validationProportion is the ratio of the dataset that is used for the 
#validation dataset.

#Convergence curve for accumulative BP algorithm process
f3 = plt.figure(3)
plt.plot(err_train, 'b', err_valid, 'r')
plt.title('BP network classification')
plt.ylabel('error rate')
plt.xlabel('epochs')
plt.show()

#testing
tst_result = percentError(trainer.testOnClassData(tstdata), 
                          tstdata['class'])
#Return percentage of mismatch between out and target values
print("epoch: %4d" % trainer.totalepochs, " test error: %5.2f%%" % tst_result)

#%%Decision tree
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.5, 
                                                    random_state = 0)
#random_state: int or RandomState
#Pseudo-random number generator state used for random sampling

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                      test_size = 0.5, 
                                                      random_state = 0)
#random_state: int or RandomState
#Pseudo-random number generator state used for random sampling

data_file_train = "./btrain.csv"
data_file_valid = "./bvalidate.csv"
data_file_test = "./btest.csv"
data_file_datatype = "./datatypes.csv"

df_train = pd.DataFrame(X_train)
df_train.columns = feature_names
#Directly use df.columns to set column names
df_train['class'] = y_train
df_train.to_csv(data_file_train)

df_valid = pd.DataFrame(X_valid)
df_valid.columns = feature_names
#Directly use df.columns to set column names
df_valid['class'] = y_valid
df_valid.to_csv(data_file_valid)

df_test = pd.DataFrame(X_test)
df_test.columns = feature_names
#Direclty use df.columns to set column names
df_test['class'] = y_test
df_test.to_csv(data_file_test)

datatypes = ['true']*df_train.shape[1]
#Use ['true']*31 to create a list with 31 elements 'true'
datatypes = pd.DataFrame(datatypes)
datatypes = datatypes.T
datatypes.to_csv(data_file_datatype, header = False)

import decision_tree

#Add command line arguments as:
#python decision-tree.py btrain.csv -v bvalidate.csv -p -t btest.csv
#This command runs decision-tree.py with btrain.csv as the training set, 
#bvalidate.csv as the validation set, 
#btest.csv as the test set, 
#and pruning enabled.
#The classifier is not specified so it defaults to the last column in the 
#training set. No datatypes file is specified so it defaults to datatypes.csv
#Printing is not enabled.
#datatypes.csv - A metadata file that indicates 
#(with comma separated true/false entries) which attributes are numeric (true) 
#and nominal (false)

decision_tree.main()

#The result -> result.csv
df_result = pd.read_csv(open('result.csv', r))
y_pred = df_result['class'].get_values()
#Use get_values() to reduce the Series to array

accuracy = metrics.accuracy_score(y_test, y_pred)
#Accuray classification score.
#normalize: bool, optional (default = True)
#If ``False``, return the number of correctly classified samples.
#Otherwise, return the fraction of correctly classified samples.

print('accuracy of C4.5 tree: %.3f' % accuracy)



    


