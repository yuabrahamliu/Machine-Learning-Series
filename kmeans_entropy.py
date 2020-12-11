# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 21:07:12 2020

@author: abrah
"""

#%%
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

Idx = Series(range(1, 31))

density = Series([0.697, 0.774, 0.634, 0.608, 0.556, 
                  0.403, 0.481, 0.437, 0.666, 0.243, 
                  0.245, 0.343, 0.639, 0.657, 0.360, 
                  0.593, 0.719, 0.359, 0.339, 0.282, 
                  0.748, 0.714, 0.483, 0.478, 0.525, 
                  0.751, 0.532, 0.473, 0.725, 0.446])

sugar_ratio = Series([0.460, 0.376, 0.264, 0.318, 0.215, 
                      0.237, 0.149, 0.211, 0.091, 0.267, 
                      0.057, 0.099, 0.161, 0.198, 0.370, 
                      0.042, 0.103, 0.188, 0.241, 0.257, 
                      0.232, 0.346, 0.312, 0.437, 0.369, 
                      0.489, 0.472, 0.376, 0.445, 0.459])

watermelon = pd.concat([Idx, density, sugar_ratio], axis = 1)
#pd.concat(objs, axis = 0) Concatenate pandas objects along a particular axis 
#                          with optional set logic along the other axes.
#objs: a sequence or mapping of Series, DataFrame, or Panel objects
#axis: The axis to concatenate along

watermelon.columns = ['Idx', 'density', 'sugar_ratio']

watermelon = watermelon.set_index(['Idx'], drop = True)
#Set the DataFrame index (row labels) using one or more existing columns. By 
#default yields a new object.

del Idx, density, sugar_ratio

#%%
class KMeans_entropy(object):
    def __init__(self, ks):
        self.ks = ks
    #During the process of writing a class, if want to test its methods being 
    #written, an annoyance is the error "NameError: name 'self' is not defined". 
    #To get a self object and use it conviently during method writing and testing, 
    #can write the __init__ method of the class first, and generate an object 
    #named 'self' using the temporary class containing the __init__ method only, 
    #like self = KMeans_entropy(range(2, 11)) here, and then this self can be 
    #used to test other methods need to be written.
    
    def find_closest_centroids(self, X, centroid): 
        """
        Retrun which clusters the samples in X belong to as an array with cluster 
        indeces for the samples (0 based number), as well as the sum of sample 
        to centroid distance within each cluster as a DataFrame
        @X: DataFrame, each row is a sample and each column is a feature
        @centroid: DataFrame, each row is a cluster centroid and each column is 
                   a feature.
        """
        Xarray = np.array(X)
        centroidarray = np.array(centroid)
        distance = np.sum((Xarray[:, np.newaxis, :] - centroidarray)**2, axis = 2)
        #np.newaxis inserts a new dimension to the original array
        #Xarray.shape is (30L, 2L). After using np.newaxis to insert a new axis 
        #to the 2nd dimension, Xarray[:,np.newaxis,:].shape is (30L, 1L, 2L)
        #Xarray[0:3,] is 
        #array([[0.697, 0.46], 
        #       [0.774, 0.376], 
        #       [0.634, 0.264]])
        #Xarray[:,np.newaxis,:][0:3,] is 
        #array([[[0.697, 0.46]], 
        #       [[0.774, 0.376]], 
        #       [[0.634, 0.264]]])
    
        #This command uses the broadcasting mechanism of numpy to calculate the 
        #3 distances of each sample to the 3 centroids (when k value is 3).
        #Xarray[:, np.newaxis, :].shape is (30L, 1L, 2L), which means it contains 
        #30 2-d arrays with a shape of (1L, 2L), not 2 2-d arrays with a shape 
        #of (30L, 1L). The 2-d array with a shape of (1L, 2L) represents 1 sample 
        #(1L) with 2 features (density, sugar_ratio) (2L)
        #centroidarray.shape is (3L, 2L) (when k value is 3)
        #Xarray[:, np.newaxis, :] - centroidarray returns an array with a shape 
        #of (30L, 3L, 2L), which means it contains 30 2-d arrays with a shape of 
        #(3L, 2L) (when k value is 3), not 2 2-d arrays with a shape of (30L, 3L). 
        #The broadcasting is performed as, for each of the 30 (1L, 2L) arrays of 
        #Xarray[:, np.newaxis, :], it will be used to minus the (3L, 2L) array 
        #of centroidarray. Although (1L, 2L) and (3L, 2L) are different shapes, 
        #because of broadcasting, the original (1L, 2L) array will be repeated 
        #for 3 times along the 1L axis and form a (3L, 2L) array, and then 
        #2 (3L, 2L) can be used to do substraction. 
        #(Xarray[:, np.newaxis, :] - centroidarray)**2 also returns an array 
        #with a shape of (30L, 3L, 2L), which means it contains 30 2-d arrays 
        #with a shape of (3L, 2L). The 2-d array with a shape of (3L, 2L) 
        #represents 3 squared Euclidean distances (3L) based on 2 features 
        #(density, sugar_ratio) (2L) between 1 sample and the 3 centroids.
        
        #np.sum(array_like, axis = None) Sum of array elements over a given axis. 
        #  axis: None or int or tuple of ints. Axis or axes along which a sum 
        #        is performed. The default, axis = None, will sum all of the 
        #        elements of the input array. If axis is a tuple of ints, a sum 
        #        is performed on all of the axes specified in the tuple instead 
        #        of a single axis or all the axes as before.
        #The final result of 
        #np.sum((Xarray[:, np.newaxis, :] - centroidarray)**2, axis = 2) has a 
        #shape of (30L, 3L), because the original object 
        #(Xarray[:, np.newaxis, :] - centroidarray)**2 has a shape of 
        #(30L, 3L, 2L), and after summing all the elements along axis 2 (axis 0 
        #is the 30L axis; axis 1 is the 3L axis; axis 2 is the 2L axis), the 
        #result should be an array with a shape of (30L, 3L), which means for 
        #the 30 samples (30L), each of them has 3 squared Euclidean distances 
        #to the 3 centroids (3L) (when k value is 3).
        
        idx = distance.argmin(axis = 1)
        #np.argmin(array_like, axis) Returns the indices of the minimum values 
        #                            along an axis.
        #The shape of distance is (30L, 3L) (when k value is 3). The final 
        #result of distance.argmin(axis = 1) is 
        #array([2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 
        #       0, 0, 2, 2, 1, 1, 2, 0, 0, 0, 
        #       2, 2, 1, 1, 1, 2, 1, 1, 2, 1], dtype = int64)
        #It means for each of the 30 samples (30L) included in distance, 
        #the minimum one of the 3 sample to centroid distances (3L) (when k 
        #value is 3), will be selected and its index will be returned. This 
        #index also indicates the cluster this sample belongs to.
        
        mindistance = distance.min(axis = 1)
        #np.min(array_like, axis) Returns the minimum of an array for minimun 
        #                         along an axis.
        mindistance = DataFrame(mindistance)
        mindistance.index = X.index
        mindistance.columns = ['distance']
        
        mindistance['cluster'] = idx
        
        clusterdistance = mindistance.groupby(by = ['cluster'], axis = 0).sum()
        #pd.groupby(by, axis = 0) by: mapping, function, str, or iterable 
        #                             Used to determine the groups for the 
        #                             groupby
        #                         axis: int, default 0
        
        clusterdistance['cluster'] = clusterdistance.index
        #The original value of the index of DataFrame clusterdistance is the 
        #value of column 'cluster' in the original DataFrame mindistance, 
        #which was used to group mindistance
        
        clusterdistance.index = range(clusterdistance.shape[0])
                
        return idx, clusterdistance
    
    def compute_centroids(self, X, idx, k): 
        """
        Return centroids of the clusters (cluster IDs are 0 based numbers)
        @X: DataFrame, each row is a sample and each column is a feature
        @idx: array with elements as cluster IDs of the samples in X
        """
        centroids = np.zeros((k, X.shape[1]))
        
        for i in range(0, k): 
            centroids[i, :] = np.mean(X.ix[idx == i], axis = 0)
            #np.mean(array_like, axis) Compute the arithmetic mean along the 
            #                          specified axis.
            #Here, idx is a 1-d array with a length of 30, and its elements are 
            #the original cluster IDs (0 based numbers) of the 30 samples. Hence, 
            #idx == i is also a 1-d array with a length of 30, and its elements 
            #are blean values indicating whether the samples belong to the 
            #original cluster with an ID of number i. 
            #The command X.ix[idx == i] uses blean value array idx == i to select 
            #rows from the DataFrame X. Because idx == i only contains blean 
            #values without numbers in the index of DataFrame X, this selection 
            #is independent of the DataFrame index value.
        
        centroids = DataFrame(centroids)
        centroids.columns = X.columns
        centroids.index = range(0, centroids.shape[0])
        
        return centroids
    
    def plot_converge(self, X, idx, initial_idx, k): 
        """
        Plot the samples in a 2-D scatter plot and label the clusters using 
        convex hulls
        @X: DataFrame, each row is a sample and each column is a feature (The 
            number of features must be 2)
        @idx: array with elements as cluster IDs of the samples in X
        @initial_idx: the initialized centroids (randomly selected samples) 
                      at the beginning of k-means clustering
        """
        plt.cla()
        #Clear the current axes
        
        plt.title('k-means converge process')
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        
        Xarray = np.array(X)
        #Create an array Xarray from the DataFrame X, and then transfer this 
        #array Xarray to plt.scatter to draw the scatter plot, rather than 
        #the DataFrame X, because plt.scatter will automatically generate 
        #a legend label for DataFrame using the last variable name, which may 
        #disturb the final legend setting step. If use the array Xarray 
        #instead, plt.scatter will not automatically generate a legend label
        
        plt.scatter(Xarray[:, 0], Xarray[:, 1], c = 'lightcoral')
        #plt.scatter(x, y) Make a scatter plot of `x` vs `y` 
        #  x, y: array_like, shape(n,)
        #  c: color, sequence, or sequence of color. default: 'b'.
        plt.scatter(Xarray[initial_idx, 0], Xarray[initial_idx, 1], 
                    label = 'initial center', c = 'k')
        
        for i in range(k): 
            X_i = Xarray[idx == i]
            #Here, idx is a 1-d array with a length of 30, and its elements are 
            #the original cluster IDs (0 based numbers) of the 30 samples. Hence, 
            #idx == i is also a 1-d array with a length of 30, and its elements 
            #are blean values indicating whether the samples belong to the 
            #original cluster with an ID of number i. 
            #The command Xarray[idx == i] uses blean value array idx == i to 
            #select rows from the array Xarray.
            
            hull = ConvexHull(X_i).vertices.tolist()
            hull.append(hull[0])
            #ConvexHull(points) Convex hulls in N dimensions
            #  points: ndarray of floats, shape(npoints, ndim)
            #          Coordinates of points to construct a convex hull form 
            #ConvexHull(points).vertices ndarray of ints, shape(nvertices,) 
            #                            Indices of points forming the vertices 
            #                            of the convex hull. 
            #                            For 2-D convex hulls, the vertices are 
            #                            in counterclockwise order.
            #array.tolist() Return the array as a (possible nested) list.
            
            plt.plot(X_i[hull, 0], X_i[hull, 1], 'c--')
            #Plot lines and/or markers
            #By default, each line is assigned a different style specified by 
            #a 'style cycle'
            #``'--'`` dashed line style
            #'c' cyan color
            
            #X_i[hull, 0] is an array. Transfer this array to plt.plot, rather 
            #than a DataFrame, because for a DataFrame, plt.plot will 
            #automatically create a legend label for it using its last variable 
            #name, which will disturb the final legend setting step. If use an 
            #array instead, plt.plot will not automatically generate a legend label
            
        plt.legend()
        #plt.legend Place a legend on the axes.
        #This method can automatically detect the elements to be shown in the 
        #legend
        #The elements to be added to the legend are automatically determined, 
        #when you do not pass in any extra arguments. 
        #In this case, the LABELs (LABEL = 'initial center' here) are taken from
        #the artist(plt.scatter(X.ix[initial_idx, 0], X.ix[initial_idx, 1], 
        #                       LABEL = 'initial center', c = 'k') here). You 
        #can specify them either at artist creation or by calling the 
        #set_label() method on the artist.
        
        plt.pause(0.5)
        #plt.pause(interval) Pause for *interval* seconds.
       
        
    def fit(self, X, k, 
            initial_centroid_index = None, max_iters = 10, seed = 16, 
            plt_process = False): 
        """
        Do k-means clustering and return the final cluster IDs of the samples 
        (0 based numbers), the cetroids of clusters, and the sum of sample to 
        centroid distance within each cluster. If set parameter plt_process = 
        True, and the number of features is 2, will also draw the samples in a 
        2-D scatter plot and label the sample clusters using convex hulls.
        @X: DataFrame, each row is a sample and each column is a feature
        """
        m, n = X.shape
        
        #If no specific centroid is assigned, initialize the centroids randomly
        if initial_centroid_index is None: 
            
            np.random.seed(seed)
            initial_centroid_index = np.random.randint(1, m+1, k)
            #randint(low, high = None, size = None)
            #Return random integers from `low` (inclusive) to `high` (exclusive). 
            #If `high` is None (the default), then results are from [0, `low`). 
            #size: int or tuple of ints 
            #      Output shape. If the given shape is, e.g., ``(m, n, k)``, 
            #      then ``m*n*k`` samples are drawn. Default is None, in which 
            #      case a single value is returned.
            
        centroid = X.ix[initial_centroid_index, :]
        centroid.index = range(centroid.shape[0])
        
        idx = None
        
        plt.ion()
        #plt.ion Turn interactive mode on
        for _ in range(max_iters): 
        #Here, _ is a dummy variable
            #Assign samples to the clusters according to the original cluster centroids, 
            #and return the sum of sample to centroid distance within each cluster
            idx, clusterdistance = self.find_closest_centroids(X, centroid)
            
            if plt_process: 
                if(X.shape[1] == 2):
                    self.plot_converge(X, idx, 
                                       initial_idx = initial_centroid_index, 
                                       k = k)    
                else: 
                    print "To draw the k-means converge process, the number \
                    of features must be 2"
                    #Note, after the line continuation character "\", there 
                    #should be NO BLANK
        
            #Calculate the new cluster centroids after sample assignment
            centroid = self.compute_centroids(X, idx, k = k)
        
        
        
        
        
        plt.ioff()
        #plt.ioff Turn interactive mode off
        
        plt.show()
        #plt.show Display a figure. When running in ipython with its pylab mode, 
        #  display all figures and return to the ipython prompt.
        
        return centroid, idx, clusterdistance
    
    def determinek(self, X, 
                   initial_centroid_index = None, max_iters = 10, seed = 16, 
                   plt_process = False, 
                   entropy_factor = 0.5): 
        """
        Determine the optimal k value by caluculating the sum of squared 
        deviation with information entropy penalized and return the optimal 
        k value, final sample cluster IDs with this k value, cluster centroids, 
        and the sum of squared deviation with information entropy penalized
        @X: DataFrame, each row is a sample and each column is a feature
        """
        score_old = np.inf
        
        for k in self.ks: 
            centroid_new, idx_new, clusterdistance_new = self.fit(X = X, 
                                                                  k = k, 
                                                                  initial_centroid_index = initial_centroid_index, 
                                                                  max_iters = max_iters, 
                                                                  seed = seed, 
                                                                  plt_process = plt_process)
            
            deviation = np.sum(clusterdistance_new.ix[:,'distance'])
            #np.sum(array_like, axis = None) Sum of array elements over a given axis. 
            #  axis: None or int or tuple of ints. Axis or axes along which a sum 
            #  is performed. The default, axis = None, will sum all of the 
            #  elements of the input array. If axis is a tuple of ints, a sum 
            #  is performed on all of the axes specified in the tuple instead 
            #  of a single axis or all the axes as before.
            
            samplesize = float(idx_new.shape[0])
            clustersize = np.unique(idx_new, return_counts = True)[1]
            #np.unique(array_like, return_counts = False) 
            #Find the unique elements of an array. 
            #Returns the SORTED unique elements of an array.
            #return_counts: If True, also return the number of times each 
            #               unique value comes up.
            entropy = np.sum((clustersize/samplesize)*np.log2(clustersize/samplesize)) 
            #np.sum(array_like, axis = None) Sum of array elements over a given axis. 
            #  axis: None or int or tuple of ints. Axis or axes along which a sum 
            #  is performed. The default, axis = None, will sum all of the 
            #  elements of the input array. If axis is a tuple of ints, a sum 
            #  is performed on all of the axes specified in the tuple instead 
            #  of a single axis or all the axes as before.
            
            score_new = deviation - entropy_factor*entropy
            #entropy is always negative, so minus entropy when the purpose is 
            #to minimize the score
            
            if score_new < score_old: 
                score_old = score_new
                optimalk = k
                idx_old = idx_new
                centroid_old = centroid_new
            else: 
                break
            
        return optimalk, idx_old, centroid_old, score_old

#%%
if __name__ == '__main__': 
    optimalk, idx, centroid, score = KMeans_entropy(range(2, 11)).determinek(
            X = watermelon, initial_centroid_index = None, max_iters = 10, 
            seed = 24, plt_process = True, entropy_factor = 0.5)          
            
#%%
%reset       
                    
            

            
            
        
        
        
        

    



