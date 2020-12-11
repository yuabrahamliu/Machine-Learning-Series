# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 19:23:56 2020

@author: abrah
"""

#%%
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

class PCA(object): 
    def __init__(self, n_components): 
        self.n_components = n_components
    
    #During the process of writing a class, if want to test its methods being 
    #written, an annoyance is the error "NameError: name 'self' is not defined". 
    #To get a self object and use it conviently during method writing and testing, 
    #can write the __init__ method of the class first, and generate an object 
    #named 'self' using the temporary class containing the __init__ method only, 
    #like self = PCA(n_components = 64) here, and then this self can be used to 
    #test other methods need to be written.
    
    def fit(self, X): 
        """
        Calculate the eigenvectors and eigenvalues of the data matrix X.
        @X: Data matrix. Each row is a sample and each column is a feature.
        """
        self.x_mean = np.mean(X, axis = 0)
        self.x_std = np.std(X, axis = 0, ddof = 0)
        #numpy.std(a, axis = None, ddof = 0) Compute the standard deviation 
        #  along the specified axis. Returns the standard deviation, a measure 
        #  of the spread of a distribution, of the array elements. The standard 
        #  deviation is computed for the flattened array by default, otherwise 
        #  over the specified axis. 
        #a: array_like Calculate the standard deviation of these values
        #axis: None or int or tuple of ints, optional Axis or axes along which 
        #  the standard deviation is computed. The default is to compute the 
        #  standard deviation of the flattened array.
        #ddof: int, optional Means Delta Degrees of Freedom. The divisor used in 
        #  calculations is  ``N - ddof``, where ``N`` represents the number of 
        #  elements. By default ``ddof`` is zero.
        
        X = (X - self.x_mean)/self.x_std
        #Use `- self.x_mean` to centralize the data, while to get rid of the 
        #influence from the variance difference of different features, which 
        #makes the result of PCA largely determined by the features with large 
        #variance, further use `/self.x_std` to standardize the data. The 
        #covariance matrix of the standardized data is actually their correlation 
        #matrix.
        
        m = X.shape[0]
        #m is the row number of the data matrix X, i.e.the sample number.
        
        sigma = np.dot(X.T, X)/m
        #numpy.dot(a, b) Dot product of two arrays. 
        #  For 2-D arrays it is equivalent to matrix multiplication, and for 1-D 
        #  arrays to inner product of vectors (without complex conjugation). 
        
        #Here, for X, each row is a sample and each column is a feature. So for 
        #X.T, each row is a feature and each column is a sample. Set the number 
        #of features as d and the number of samples as m, then X is a mxd matrix, 
        #and X.T is a dxm matrix. Hence, for np.dot(X.T, X), it is a dxd matrix. 
        #Actually, sigma = np.dot(X.T, X)/m is the correlation matrix of the 
        #standardized data (because the standard deviation used to standardize 
        #the matrix is calculated via dividing the sample number m directly, here 
        #the divisor is correspondingly `/m`, but strictly, the standard deviation 
        #should be calcualted via dividing the degree of freedom (m-1), and 
        #hence the divisor to calculate the correlation matrix should also be 
        #`/(m-1)`, rather than `/m`). While if X is just a centralized data 
        #matrix, sigma is the covariance matrix.
        
        vals, vecs = np.linalg.eig(sigma)
        #numpy.linalg.eig(a) Compute the eigenvalues and right eigenvectors of 
        #  a square array.
        #a: (..., M, M) array MatriCES for which the eigenvalues and right 
        #  eigenvectors will be computed
        #Returns 
        #w: (..., M) array The eigenvalues, each repeated according to its 
        #  multiplicity. 
        #v: (..., M, M) array The normalized (unit "length") eigenvectors, such 
        #  that the COLUMN ``v[:i]`` is the eigenvector corresponding to the 
        #  eigenvalue ``w[i]``.
        
        self.vals = vals
        self.vecs = vecs
        
        return self
    
    def transform(self, X): 
        """
        Reduce the dimension for the samples
        @X: The original data matrix. Each row is a sample and each column is a 
            feature.
        """
        return np.dot(X, self.vecs[:,:self.n_components])
        #numpy.dot(a, b) Dot product of two arrays. 
        #  For 2-D arrays it is equivalent to matrix multiplication, and for 1-D 
        #  arrays to inner product of vectors (without complex conjugation). 
        
        #Here, for X, each row is a sample and each column is a feature. Set the 
        #number of samples as m and the number of features as d, then X is a mxd 
        #matrix. In addition, self.vecs[:,:self.n_components] is the projection 
        #matrix, with d rows and d' columns (d' <= d). For each of its columns, 
        #it is a dx1 vector that keeps the original dimension number of d. In the 
        #original dxd space, the number of such dx1 column vectors is d, but 
        #after PCA, only d' of them are kept, and the dimension reduction 
        #is reflected by the reduction of the number of the dx1 vectors from 
        #d to d', but for each vector, it still contains d elements, so the 
        #projection matrix still has d rows, and its shape is dxd'. The dot 
        #between the original data matrix X (mxd) and the projection matrix 
        #(dxd') is a mxd' matrix, which is the new data matrix after dimension 
        #reduction. For this matrix, each of its row is a sample and each of 
        #its column is a feature, so from this matrix, it can be seen that 
        #after PCA, the sample number is still the same as m, but for each sample, 
        #its feature number is reduced from the orignal d to d'. 
        
    def recover(self, Z): 
        """
        Recover the original data matrix from the reduced data matrix and 
        the projection matrix, to faciliate the comparison between the diplays 
        before and after PCA
        @Z: The reduced data matrix. Each row is a sample and each column is a 
            feature.
        """
        return np.dot(Z, self.vecs[:,:self.n_components].T)
        #numpy.dot(a, b) Dot product of two arrays. 
        #  For 2-D arrays it is equivalent to matrix multiplication, and for 1-D 
        #  array to inner product of vectors (without complex conjugation). 
        
        #Here, for Z, each row is a sample and each column is a feature (after 
        #PCA). Set the number of samples as m and the number of features (after 
        #PCA) as d', then Z is a mxd' matrix. In addition, 
        #self.vecs[:,:self.n_components] is the original projection matrix, and 
        #self.vecs[:,:self.n_components].T here is its transpose matrix. For the 
        #original projection matrix, it has d rows and d' columns (d' <= d). For 
        #each of its columns, it is a dx1 vector that keeps the original 
        #dimension number of d. In the original dxd space, the number of such dx1 
        #column vectors is d, but after PCA, only d' of them are kept, and the 
        #dimension reduction is reflected by the reduction of the number of the 
        #dx1 vectors from d to d', but for each vector, it still contains d 
        #elements, so the projection matrix still has d rows, and its shape is 
        #dxd'. Correspondingly, the transpose matrix 
        #self.vecs[:,:self.n_components].T has a shape of d'xd, with d' rows and 
        #d columns. The dot between the reduced data matrix Z (mxd') and the 
        #transpose matrix (d'xd) is a mxd matrix, which is the original data 
        #matrix recovered. For this matrix, each of its row is a sample and 
        #each of its column is a feature, so for this matrix, the sample number 
        #is still the same as m, but for each sample, its feature number is 
        #recovered to the original value of d, from the reduced value of d'.
        
        #Previously, the reduced data matrix Z was obtained via the dot computation 
        #of Z = np.dot(X, self.vecs[:,:self.n_components]), where X is the original 
        #data matrix. Here, the recovered data matrix (name it as X_hat) is 
        #obtained via X_hat = np.dot(Z, self.vecs[:,:self.n_components].T). Hence, 
        #one can go directly from X to X_hat with the matrix 
        #np.dot(self.vecs[:,:self.n_components], self.vecs[:,:self.n_components].T). 
        #The column number of the matrix self.vecs[:,:self.n_components] is 
        #self.n_components (i.e. d'), indicating the number of eigienvectors involved 
        #in it is self.n_components (i.e. d'). If d' == d, i.e. if all d eigenvectors 
        #are used, the matrix np.dot(self.vecs[:,:self.n_components], \
        #self.vecs[:,:self.n_components].T) is the identity matrix (no 
        #dimensionality reduction is performed, hence "reconstruction" is prefect). 
        #If only a subset of eigenvectors is used, it is not identity, and there 
        #are some differences between X_hat and the orignal X.
        

    
def display_facedata(X_, title = None): 
    """
    Show the image from the image matrix.
    X_: Image matrix. Each row is a sample and each column is a feature. Each 
        feature is a tile of the sample image. These tiles pile up one another 
        and constitute a square. The square is the final image. To display the 
        tiles in a square, rather than in one row as recorded in the image matrix 
        X_, the tiles need to be expanded into a square via sqrt.
    title: Title of the image
    """
    m, n = X_.shape
    display_rows = 10
    display_cols = int(m/10)
    #m is the row number of the image matrix X_, i.e. the sample number. Here, 
    #desolve m into display_rows and display_cols, 
    #so that m = display_rows * display_cols, meaning the samples recorded in 
    #X_ will finally be shown in a matrix, with each row containing display_rows 
    #samples and each column containing display_cols samples, not just in one 
    #long column
    
    example_hight = int(np.sqrt(n))
    #n is the column number of the image matrix X_, i.e. the feature number. 
    #Each sample has n features and each feature is a tile of the final sample 
    #image. These tiles pile up one another and constitute a square. The square 
    #is the final image. To display the tiles in a square, rather than in one 
    #row as recorded in the image matrix, the tiles need to be expanded into 
    #a square and the length of the square edge should be 
    #example_hight = int(np.sqrt(n)).
    
    display_face = np.zeros((example_hight * display_rows, \
                             example_hight * display_cols))
    #Note, after the line continuation character "\", there should be NO BLANK
    
    #zeros(shape) Return a new array of given shape and type, filled with zeros. 
    #shape: int or sequence of ints Shape of the new array, e.g., ``(2, 3)`` 
    #    or ``2``.
    
    temp_m = 0
    for i in range(display_rows): 
        for j in range(display_cols): 
            display_face[i * example_hight: (i + 1) * example_hight, \
                         j * example_hight: (j + 1) * example_hight] = \
                         X_[temp_m, :].reshape(example_hight, -1).T
            #reshape(a, newshape) Gives a new shape to an array without changing 
            #  its data.
            #a: array_like Array to be reshaped.
            #newshape: int or tuple of ints The new shape should be compatible 
            #  with the original shape. If an integer, then the result will be 
            #  a 1-D array of that length. One shape dimension can be -1. In 
            #  this case, the value is inferred from the length of the array and 
            #  remaining dimensions.
            
            temp_m += 1
    
    plt.title(title)
    #plt.title Set a title of the current axes. 
    #Set one of the three available axes titles. The available titles are 
    #positioned above the axes in the center, flush with the left edge, 
    #and flush with the right edge.
    
    plt.imshow(display_face)
    #imshow(X, norm = None, cmap = None) Display an image on axes. 
    #X: array_like, shape (n, m) or (n, m, 3) or (n, m, 4) 
    #  Display the image in `X` to current axes. `X` may be an array or a PIL 
    #  image. If `X` is an array, it can have the following shapes and types: 
    #  - MxN -- values to be mapped (float or int)
    #  - MxNx3 -- RGB (float or unit8)
    #  - MxNx4 -- RGBA (float or unit8)
    #  The value of each component of MxNx3 and MxNx4 float arrays should be in 
    #  the range 0.0 to 1.0. MxN arrays are mapped to colors based on the `norm` 
    #  (mapping scalar to scalar) and the `cmap` (mapping the normed scalar to 
    #  a color). 
    #norm: `~matplotlib.colors.Normalize`, optional, default: None 
    #  A `~matplotlib.colors.Normalize` instance is used to scale a 2-D float 
    #  `X` input to the (0, 1) range for input to the `cmap`. If `norm` is None, 
    #  use the default func:`normalize`.
    #cmap: `~matplotlib.colors.Colormap`, optional, default: None 
    #  If None, default to rc `image.cmap` value. `cmap` is ignored if `X` is 
    #  3-D, directly specifying RGB(A) values.
    
    plt.show()
    #plt.show Display a figure. When running in ipython with its pylab mode, 
    #  display all figures and return to the ipython prompt.

#%%
if __name__ == '__main__': 
    data = scio.loadmat(r'D:/machine_learning/ex7faces.mat')
    #loadmat(file_name) Load MATLAB file. 
    #file_name: str Name of the mat file. Can also pass open file-like object.
    
    X_face = data['X']
    #data is a dictionary. In the key 'X', there is the image matrix.


#Display the image before dimension reduction
display_facedata(X_ = X_face[:100, :], title = "original face")

#Reduce the dimension from the original 1024 to 64
pca = PCA(64)

Z_face = pca.fit(X_face).transform(X_face)

Z_recover = pca.recover(Z_face)

#Display the image after dimension reduction
display_facedata(X_ = Z_recover[:100, :], title = "after the dimension reduction")

#Display the top 100 eigenvectors
display_facedata(pca.vecs[:,:100].T, title = "eigenvector")
#The original purpose of PCA is to find an orthonormal set {vec_1, vec_2, ..., 
#vec_d'}, to minimize E_d'({vec_j}) = Sigma ||(mean + Sigma(Z_kj * vec_j)) - X_k||^2, 
#where j = 1 to d' (d' is the dimension number after reduction) and k = 1 to m 
#(m is the sample number), and d' <= d (d is the original dimension number), i.e., 
#use Sigma(Z_kj * vec_j) to approximate X_k - mean, i.e., 
#X_k - mean = Z_k1 * vec_1 + Z_k2 * vec_2 + Z_k3 * vec_3 + ... + Z_kd' * vec_d' 
#approximately. Hence, it can be seen that for the principle components (or the 
#unit eigenvectors) vec_1, vec_2, ..., vec_d', their function is that, 
#their weighted sum can be used to reconstruct the centered sample X_k - mean 
#approximately (the weights are the principle component coefficients or scores, 
#Z_k1, Z_k2, ..., Z_kd'), i.e., each sample X_k can be expressed as the weighted 
#sum of the eigenvectors (X_k is a dx1 vector, Z_k1, Z_k2, ..., Z_kd' are all 
#scalar number, and vec_1, vec_2, ..., vec_d' are also dx1 vectors). 

#In the case of image dimension reduction here, each sample X_k is a face with 
#1024 features (tiles) originally, and each eigenvector (named as "eigenface" 
#here) is also a face image with 1024 features (tiles), and the sample face X_k 
#can be expressed as the weighted sum of the 100 eigenfaces, i.e., the weighted 
#overlap of the 100 eigenface images can finally generate the sample face X_k 
#image. Different sample face has a different set of weights (principle component 
#scores), so via overlap from the same set of 100 eigenface images, different 
#sample face images can be obtained. Here, use only 100 eigenvectors (eigenfaces), 
#rather than the whole 1024 eigenvectors (eigenfaces) can generate the original 
#sample faces exactly, because all the remaining 1024 - 100 = 924 eigenvectors 
#have an eigenvalue of 0, due to the fact that the original image data matrix 
#X_ as input has a shape of 100x1024 (100 samples as rows and 1024 features as 
#columns) , so the centered data matrix (set as X here) also has a shape of 
#100x1024, and hence its rank is not greater than min(100, 1024) = 100. 
#Futhermore, the matrix np.dot(X.T, X) also has a rank <= 100, so for its 
#eigenvalue decomposition, although it can get a diagonal matrix Delta with 1024 
#eigenvalues, only <= 100 of them are greater than 0, and correspondingly, besides 
#the top 100 eigenvectors, all the other 1024 - 100 = 924 eigenvectors have a 
#eigenvalue of 0 and actually will not play a role in PCA. So only use the top 
#100 eigenvectors (eigenfaces) here can reconstruct the original sample faces 
#exactly.

#For the above X_k - mean = Z_k1 * vec_1 + Z_k2 * vec_2 + ... + Z_kd' * vec_d', 
#it can be written as a matrix computation form as 
#X_k - mean = [vec_1, vec_2, ..., vec_d'] * Z_k approximately, where X_k is a 
#dx1 vector, indicating the kth original sample X_k, Z_k is a d'x1 vector, and 
#its elements are the corresponding principle component coefficients (scores) 
#of the eigenvectors vec_1, vec_2, ..., vec_d', i.e. Z_k1, Z_k2, ..., Z_kd'. 
#All of vec_1, vec_2, ..., vec_d' are dx1 vectors and they are eigenvectors. 
#If set W* = [vec_1, vec_2, ..., vec_d'], the above formula can be further 
#written as X_k - mean = W* * Z_k approximately, and if set 
#X = [X_1 - mean, X_2 - mean, ..., X_m - mean], and set 
#Z = [Z_1, Z_2, ..., Z_m], the above formalu can be extended as 
#X = W* * Z = X_hat approximately. The shape of X is dxm, that of W* is dxd', 
#that of Z is d'xm, and that of X_hat, which is the approximately constructed 
#data matrix, is dxm.


#%%
%reset
