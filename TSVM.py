# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 00:22:53 2021

@author: liuy47
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

data, label = iris.data[50:, :], iris.target[50:] * 2 - 3

#standardizing
sc = StandardScaler()
#Standardize features by removing the mean and scaling to unit variance

sc.fit(data)
#sc.fit(X)
#Compute the mean and std to be used for later scaling.
#Parameters
#X: {array-like, sparse matrix}, shape [n_samples, n_features]
#  The data used to compute the mean and standard deviation used for later 
#  scaling along the features axis.

data = sc.transform(data)
#sc.transform(X)
#Perform standardization by centering and scaling
#Parameters
#X: {array-like, sparse matrix}, shape [n_samples, n_features]
#  The data used to scale along the features axis.

test_d, test_c = np.concatenate((data[:15], data[50:65])), \
    np.concatenate((label[:15], label[50:65]))
#concatenate((a1, a2, ...), axis = 0)
#Join a sequence of arrays along an existing axis.
#Parameters
#a1, a2, ...: sequence of array_like
#axis: int The axis along which the arrays will be joined.

l_d, l_c = np.concatenate((data[45:50], data[95:])), \
    np.concatenate((label[45:50], label[95:]))
#concatenate((a1, a2, ...), axis = 0)
#Join a sequence of arrays along an existing axis.
#Parameters
#a1, a2, ...: sequence of array_like
#axis: int The axis along which the arrays will be joined.

u_d = np.concatenate((data[15:45], data[65:95]))
#concatenate((a1, a2, ...), axis = 0)
#Join a sequence of arrays along an existing axis.
#Parameters
#a1, a2, ...: sequence of array_like
#axis: int The axis along which the arrays will be joined.

lu_d = np.concatenate((l_d, u_d))

n = len(l_d) + len(u_d)

clf0 = svm.SVC(C = 1, kernel = 'linear')
clf0.fit(l_d, l_c)

lu_c_0 = clf0.predict(lu_d)

clf1 = svm.SVC(C = 1, kernel = 'linear')
clf1.fit(l_d, l_c)

u_c_new = clf1.predict(u_d)

cu, cl = 0.001, 1
sample_weight = np.ones(n)
sample_weight[len(l_c):] = cu
id_set = np.arange(len(u_d))
#arange([start,] stop[, step,])
#Return evenly spaced values within a given interval
#Values are generated within the half-open interval ``[start, stop)``
#For integer arguments the function is equivalent to the Python built-in 
#`range` function, but returns an ndarray rather than a list.

while cu < cl: 
    lu_c = np.concatenate((l_c, u_c_new))
    clf1.fit(lu_d, lu_c, sample_weight = sample_weight)
    #clf1.fit(X, y, sample_weight = None)
    #Fit the SVM model according to the given training data.
    #Parameters
    #sample_weight: array-like of shape (n_samples,), default = None 
    #  Per-sample weights. Rescale C per sample. Higher weights force 
    #  the classifier to put more emphasis on these points.
    while True: 
        u_c_new = clf1.predict(u_d)
        u_dist = clf1.decision_function(u_d)
        #clf1.decision_function(X)
        #Evaluates the decision function for the samples in X.
        #Parameters
        #X: array-like of shape (n_shape, n_features)
        #Returns
        #X: ndarray of shape (n_samples, n_class*(n_class-1)/2)
        #  Returns the decision function of the sample for each class in the 
        #  model.
        #  If in svm.SVC, the parameter decision_funcition_shape = 'ovr', the 
        #  shape is (n_samples, n_classes).
        #If in svm.SVC, the parameter decision_function_shape = 'ovo', the function 
        #values are proportional to the distance of the samples X to the separating 
        #hyperplane. If the exact distances are required, divide the function 
        #values by the norm of the weight vector (``coef_``). 
        #If in svm.SVC, the parameter decision_function_shape = 'ovr' (default), 
        #the decision function is a monotonic transformation of ovo decision 
        #function.
        
        #dist_i = y_i*(X_i*w^T/||w|| + b/||w||), where
        #dist_i is the absolute distance between sample X_i and the separating 
        #hyperplane (if X_i is on the correct side of the hyperplane, the 
        #distance is positive, otherwise, it is negative)
        #X_i*w^T + b is the decision function
        
        #dist_i' = X_i*w^T/||w|| + b/||w||, where 
        #dist_i' is the signed distance between sample X_i and the separating 
        #hyperplane
        #X_i*w^T + b is the decision function
        norm_weight = np.linalg.norm(clf1.coef_)
        epsilon = 1 - u_c_new*u_dist
        #epsilon should be loss(u_c_new*u_dist)
        #If the loss function is hinge loss, epsilon = max(0, 1 - u_c_new*u_dist), 
        #so that epsilon is always >= 0.
        #For epsilon = 1 - uc_new*u_dist, without loss function transformation, 
        #epsilon can be < 0, so if the original epsilon == 0, it will be < 0 
        #here, but because it does not influence the case that original epsilon 
        #> 0, it also does not influence the downstream steps because they only 
        #work when orignal epsilon > 0, not == 0.
        
        plus_set, plus_id = epsilon[u_c_new > 0], id_set[u_c_new > 0]
        minus_set, minus_id = epsilon[u_c_new < 0], id_set[u_c_new < 0]
        plus_max_id, minus_max_id = plus_id[np.argmax(plus_set)], \
            minus_id[np.argmax(minus_set)]
        a, b = epsilon[plus_max_id], epsilon[minus_max_id]
        
        if a > 0 and b > 0 and a + b > 2: 
            u_c_new[plus_max_id], u_c_new[minus_max_id] = \
                -u_c_new[plus_max_id], -u_c_new[minus_max_id]
            lu_c = np.concatenate(l_c, u_c_new)
            clf1.fit(lu_d, lu_c, sample_weight=sample_weight)
        else: 
            break
    cu = min(cu * 2, cl)
    sample_weight[len(l_c):] = cu

lu_c = np.concatenate((l_c, u_c_new))
test_c1 = clf0.predict(test_d)
test_c2 = clf1.predict(test_d)
score1 = clf0.score(test_d, test_c)
#clf0.score(X, y)
#Return the mean accuracy on the given test data and labels.
#Parameters
#X: array-like of shape (n_samples, n_features) Test samples.
#y: array-like of shape (n_samples,) or (n_samples, n_outputs) True labels for X.
score2 = clf1.score(test_d, test_c)
#clf1.score(X, y)
#Return the mean accuracy on the given test data and labels.
#Parameters
#X: array-like of shape (n_samples, n_features) Test samples.
#y: array-like of shape (n_samples,) or (n_samples, n_outputs) True labels for X.

fig = plt.figure(figsize = (16, 4))
#Create a new figure, or activate an existing figure.
ax = fig.add_subplot(131)
#fig.add_subplot(*args)
#*args: int, (int, int, *index*), default: (1, 1, 1)
#  The position of the subplot describe by one of 
#  -Three integers (*nrows*, *ncols*, *index*). The subplot will take the *index* 
#   position on a grid with *nrows* rows and *ncols* columns. *index* starts at 
#   1 in the upper left corner and increases to the right. *index* can also be 
#   a two-tuple specifying the (*first*, *last*) indices (1-based, and including 
#   *last*) of the subplot, e.g., ``fig.add_subplot(3, 1, (1, 2))`` makes a 
#   subplot that spans the upper 2/3 of the figure.
#  -A 3-digit integer. The digits are interpreted as if given separately as three 
#   single-digit integers, i.e. ``fig.add_subplot(235)`` is the same as 
#   ``fig.add_subplot(2, 3, 5)``. Note that this can only be used if there are 
#   no more than 9 subplots. 
ax.scatter(test_d[:,0], test_d[:,2], c = test_c, marker = 'o', cmap = plt.cm.coolwarm)
#Parameters
#c: array-like or list of colors or color, optional
#   The marker colors. Possible values:
#   - A scalar or sequence of n numbers to be mapped to colors using *cmap* and 
#     *norm*
#   - A 2-D array in which the rows are RGB or RGBA. 
#   - A sequence of colors of length n.
#   - A single color format string.
ax.set_title('True Labels for test samples', fontsize = 16)

ax1 = fig.add_subplot(132)
#fig.add_subplot(*args)
#*args: int, (int, int, *index*), default: (1, 1, 1)
#  The position of the subplot describe by one of 
#  -Three integers (*nrows*, *ncols*, *index*). The subplot will take the *index* 
#   position on a grid with *nrows* rows and *ncols* columns. *index* starts at 
#   1 in the upper left corner and increases to the right. *index* can also be 
#   a two-tuple specifying the (*first*, *last*) indices (1-based, and including 
#   *last*) of the subplot, e.g., ``fig.add_subplot(3, 1, (1, 2))`` makes a 
#   subplot that spans the upper 2/3 of the figure.
#  -A 3-digit integer. The digits are interpreted as if given separately as three
#   single-digit integers, i.e. ``fig.add_subplot(235)`` is the same as 
#   ``fig.add_subplot(2, 3, 5)``. Note that this can only be used if there are 
#   no more than 9 subplots.
ax1.scatter(test_d[:,0], test_d[:,2], c = test_c1, marker = 'o', cmap = plt.cm.coolwarm)
#Parameters
#c: array-like or list of colors or color, optional
#   The marker colors. Possible values: 
#   - A scalar or sequence of n numbers to be mapped to colors using *cmap* and 
#     *norm*
#   - A 2-D array in which the rows are RGB or RGBA.
#   - A sequence of colors of length n.
#   - A single color format string.
ax1.scatter(lu_d[:,0], lu_d[:,2], c = lu_c_0, marker = 'o', s = 10, cmap = plt.cm.coolwarm, 
            alpha = 0.6)
ax1.set_title('SVM, score: {0:.2f}%'.format(score1*100), fontsize = 16)

ax2 = fig.add_subplot(133)
#fig.add_subplot(*args)
#*args: int, (int, int, *index*), default: (1, 1, 1)
#  The position of the subplot describe by one of 
#  -Three integers (*nrows*, *ncols*, *index*). The subplot will take the *index*
#   position on a grid with *nrows* rows and *ncols* columns. *index* starts at
#   1 in the upper left corner and increases to the right. *index* can also be 
#   a two-tuple specifying the (*first*, *last*) indices (1-based, and including 
#   *last*) of the subplot, e.g., ``fig.add_subplot(3, 1, (1, 2))`` makes a 
#   subplot that spans the upper 2/3 of the figure.
#  -A 3-digit integer. The digits are interpreted as if given separately as three
#   single-digit intgers, i.e. ``fig.add_subplot(235)`` is the same as 
#   ``fig.add_subplot(2, 3, 5)``. Note that this can only be used if there are 
#   no more than 9 subplots.
ax2.scatter(test_d[:,0], test_d[:,2], c = test_c2, marker = 'o', cmap = plt.cm.coolwarm)
#Parameters
#c: array-like or list of colors or color, optinal
#   The marker colors. Possible values: 
#   - A scaler or sequence of n numbers to be mapped to colors using *cmap* and
#     *norm*
#   - A 2-D array in which the rows are RGB or RGBA.
#   - A sequence of colors of length n.
#   - A single color format string.
ax2.scatter(lu_d[:,0], lu_d[:,2], c = lu_c, marker = 'o', s = 10, cmap = plt.cm.coolwarm, 
            alpha = 0.6)
ax2.set_title('TSVM, score: {0:.2f}%'.format(score2*100), fontsize = 16)

for a in [ax, ax1, ax2]:
    a.set_xlabel(iris.feature_names[0])
    a.set_ylabel(iris.feature_names[2])

plt.show()











    
    
    
    
    









