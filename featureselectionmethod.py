'''
Feature Selection Methods for Machine Learning
This module implements a collection of feature selection algorithms for machine learning tasks, including:
- Minimum Representation Error Feature Selection (`min_representation_err_fs`)
- Kernel PCA-based Feature Selection (`kpca_feature_selection`)
- Multi-Cluster Feature Selection via Spectral Clustering and LARS (`multi_cluster_fs`)
- Mutual Information and Cosine Similarity Index Feature Selection (`mici_fs`)
- Mutual Correlation Feature Selection (`mutual_corre_fs`)
- Maximum Determinant Covariance Minimization Feature Selection (`mdcm_fs`)
- Linear Dependency Minimization Feature Selection (`linear_dependency_fs`)
- Minimum Projection Minimum Redundancy Feature Selection (`mpmr_fs`)
- Robust Sparse PCA Feature Selection (`rs_pca`)
Utility functions for internal use:
- `_copmute_rep_error`: Computes representation error for selected features.
- `_Eigenmap`: Laplacian Eigenmap embedding for spectral methods.
- `_Go`, `_IPU`: Internal optimization routines for robust sparse PCA.
Each method provides a different approach to selecting a subset of features from a data matrix, either supervised or unsupervised, based on criteria such as reconstruction error, spectral embedding, mutual correlation, redundancy minimization, or robust projections.
Dependencies:
    - numpy
    - scipy
    - scikit-learn
Functions:
    min_representation_err_fs(X, y, num_selected_feature, correlation_matrix=None, searchmode=0)
        Selects features by minimizing representation error.
    multi_cluster_fs(X, num_selected_features=-1, y=None, laplacian_reduction_dim=5, W=None)
        Multi-cluster feature selection via spectral embedding and LARS regression.
    mici_fs(X, num_selected_feature, similarity_matrix=None)
        Feature selection based on mutual information and cosine similarity.
    mutual_corre_fs(X, num_selected_feature, similarity_matrix=None)
        Feature selection maximizing mutual correlation among selected features.
    mdcm_fs(X, num_selected_feature, covariance_matrix=None)
        Feature selection maximizing the determinant of the covariance submatrix.
    linear_dependency_fs(X, num_selected_feature, covariance_matrix=None)
        Feature selection minimizing linear dependency among features.
    mpmr_fs(X, num_selected_feature=-1, correlatin_matrix=None)
        Minimum projection minimum redundancy feature selection.
    rs_pca(X, sigma, num_selected_feature)
        Robust sparse PCA-based feature selection.
Internal/Utility Functions:
    _copmute_rep_error(X, y, selected_cols=None, correlation_matrix=None)
        Computes the Frobenius norm error for a linear representation.
    _Eigenmap(W, ReducedDim=10, bEigs=None)
        Computes Laplacian Eigenmap embedding.
    _Go(A, m, k)
        Internal optimization for robust sparse PCA.
    _IPU(A, d, m, k, W0=None)
        Internal iterative projection update for robust sparse PCA.
    - All random processes are seeded for reproducibility.
    - Most algorithms use iterative greedy or projection-based optimization.
    - Some methods support both supervised and unsupervised modes.
    - Early stopping is used when convergence is detected.
'''
import numpy as np
import math
from scipy.linalg import pinv, eigh,qr
from scipy.sparse.linalg import eigs,eigsh
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Lars, LassoLars

from scipy.sparse import diags, csr_matrix, issparse
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity,rbf_kernel


def _copmute_rep_error(X, y, selected_cols = None, correlation_matrix = None):
    """
    Computes the representation error of a linear model using selected features.
    This function calculates the error between the target vector `y` and its linear approximation
    using a subset of columns from the feature matrix `X`. The error is measured as the L2 norm
    of the residuals after fitting the model. Optionally, a precomputed correlation matrix can be provided.
    Args:
        X (np.ndarray): The feature matrix of shape (n_samples, n_features).
        y (np.ndarray): The target vector of shape (n_samples,).
        selected_cols (list or np.ndarray, optional): Indices of columns in `X` to use for the model.
            If None, all columns are used.
        correlation_matrix (np.ndarray, optional): Precomputed correlation matrix of shape
            (n_features, n_features). If provided and `selected_cols` is not None, the relevant
            submatrix is used. If None, the correlation matrix is computed from `X_s`.
    Returns:
        float: The L2 norm of the residuals (representation error) after fitting the model.
    """
    
    if(selected_cols is not None):
        X_s = X[:, selected_cols]
    else:
        X_s = X
    #X_s = X_s - np.mean(X_s, axis = 0)#去均值
    if(correlation_matrix is None):
        Cor_matrix = X_s.T @ X_s
    else:
        Cor_matrix = correlation_matrix[np.ix_(selected_cols, selected_cols)]
    w = np.linalg.pinv(Cor_matrix) @ X_s.T @ y
    # 3. 计算残差矩阵和误差
    R = y - X_s @ w  
    error = np.linalg.norm(R, ord = 2)  # Frobenius 范数
    return error
   

def min_representation_err_fs(X, y, num_selected_feature, correlation_matrix = None, searchmode = 0):
    """
    Selects a subset of features from X that minimizes the linear representation error with respect to y.
    This function implements two search modes for feature selection:
    1. Local search (searchmode=0): Iteratively swaps features to minimize the representation error.
    2. Greedy forward selection (searchmode=1): Adds features one by one, each time selecting the feature that most reduces the error.
    Parameters
    ----------
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target vector of shape (n_samples,).
    num_selected_feature : int
        The number of features to select.
    correlation_matrix : np.ndarray, optional
        Precomputed feature correlation matrix (X.T @ X). If None, it will be computed internally.
    searchmode : int, optional
        The search strategy to use:
            0 - Local search with random initialization and iterative improvement (default).
            1 - Greedy forward selection.
    Returns
    -------
    selected_indices : list of int
        Indices of the selected features.
    err_new : float
        The final representation error after feature selection.
    Notes
    -----
    - The function relies on an internal helper `_copmute_rep_error` to compute the representation error.
    - For reproducibility, random seed is set to 42 in local search mode.
    """
    
    _, n_featrues = X.shape
    if(correlation_matrix == None):
        correlation_matrix = X.T @ X
    ######### Local search with random initialization and iterative improvement #########
    if (searchmode ==0):
        np.random.seed(42)
        selected_indices = np.sort(np.random.choice(n_featrues, size = num_selected_feature, replace = False))        
        iter_times = 10
        for iter in range(iter_times):
            err_old = _copmute_rep_error(X, y, selected_indices,correlation_matrix)
            err_new = err_old
            for i in range(num_selected_feature):
                for j in range(n_featrues):
                    if (np.isin(j, selected_indices)):
                        continue
                    selected_indices_tmp = selected_indices.copy()
                    selected_indices_tmp[i] = j
                    err_tmp = _copmute_rep_error(X, y, selected_indices_tmp, correlation_matrix)
                    if(err_tmp < err_new):
                        err_new = err_tmp
                        selected_indices[i] = j
            if(np.abs(err_new - err_old) < 1e-6):
                break
    ############## Greedy forward selection ##########
    if(searchmode == 1):
        #########  the first feature #########
        err = y.sum()
        first_ind = 0
        for i in range(n_featrues):
            cur_feature = X[:,i]
            w = cur_feature.T @ y / (cur_feature.T @ cur_feature)
            err_tmp = np.linalg.norm(cur_feature * w  - y, ord = 2)
            if(err_tmp < err):
                err = err_tmp
                first_ind = i
        selected_indices =[first_ind]
        err_new = err
        num_selected_current = 1
        print(f'first feature ind = {first_ind}')
        
        ######### main loop #########
        while (num_selected_current < num_selected_feature):
            add_feature_index = 0
            for i in range(n_featrues):
                if (np.isin(i, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp.append(i)
                err_tmp = _copmute_rep_error(X, y, selected_indices_tmp, correlation_matrix)
                if(err_tmp < err_new):
                    err_new = err_tmp
                    add_feature_index = i
            selected_indices.append(add_feature_index)
            num_selected_current += 1
            
    return selected_indices, err_new


def _Eigenmap(W, ReducedDim = 10, bEigs = None):
    """
    Perform eigenmap dimensionality reduction using the normalized Laplacian of the affinity matrix.
    This function computes the eigenmap embedding of the input affinity matrix `W` by performing
    eigendecomposition on its normalized Laplacian. It supports both dense and sparse matrices,
    and automatically chooses between dense and sparse eigendecomposition based on the matrix size.
    Parameters
    ----------
    W : numpy.ndarray or scipy.sparse matrix, shape (n_samples, n_samples)
        Affinity (similarity) matrix. Must be square (n_samples x n_samples).
    ReducedDim : int, optional (default=10)
        Target number of dimensions for the embedding (excluding the trivial first eigenvector).
    bEigs : bool or None, optional (default=None)
        Whether to use sparse eigendecomposition (`eigsh`). If None, automatically determined
        based on matrix size and target dimension.
    Returns
    -------
    Y : numpy.ndarray or scipy.sparse matrix, shape (n_samples, ReducedDim)
        The embedding of the data in the reduced dimensional space (excluding the first trivial eigenvector).
    eigvals : numpy.ndarray, shape (ReducedDim,)
        The eigenvalues corresponding to the embedding dimensions (excluding the first trivial eigenvalue).
    Raises
    ------
    ValueError
        If the input matrix `W` is not square.
    Notes
    -----
    - The function normalizes the Laplacian using the degree matrix.
    - The first eigenvector (corresponding to the trivial solution) is omitted from the output.
    - Handles both dense and sparse affinity matrices.
    """
  
    nSmp = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError('W必须是方阵!')
    
    # default parameters
    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1
    ReducedDim = min(ReducedDim + 1, nSmp)
    
    # ========== 2. normalized Laplacian matrix ========== [5,8](@ref) 
 
    D = np.array(W.sum(axis=1)).flatten()
    D_mhalf = D ** -0.5
    D_mhalf[~np.isfinite(D_mhalf)] = 0 
    

    D_mhalf_diag = diags(D_mhalf, format='csr')
    if issparse(W):
        W_norm = D_mhalf_diag.dot(W).dot(D_mhalf_diag)
    else:
        W_norm = D_mhalf_diag @ W @ D_mhalf_diag
    

    W_norm = np.maximum(W_norm, W_norm.T) if not issparse(W) else (W_norm + W_norm.T) / 2
    
    # ========== 3. eigenvalue decomposition strategy ========== 
    if bEigs is None:
        bEigs = (nSmp > MAX_MATRIX_SIZE) and (ReducedDim < nSmp * EIGVECTOR_RATIO)
    
    # ========== 4. eigenvalue decomposition ==========
    if bEigs:
        eigvals, eigvecs = eigsh(
            W_norm, 
            k = ReducedDim, 
            #sigma = 0.01,
            which = 'LM',  # 
            tol= 1e-6
        )
        eigvals = eigvals[::-1]  #
        eigvecs = eigvecs[:, ::-1]
    else:
       
        if issparse(W_norm):
            W_norm = W_norm.toarray()
        eigvals, eigvecs = eigh(W_norm)
        eigvals = eigvals[::-1]  # 
        eigvecs = eigvecs[:, ::-1]
        

        if ReducedDim < len(eigvals):
            eigvals = eigvals[:ReducedDim]
            eigvecs = eigvecs[:, :ReducedDim]
    

    valid_idx = np.abs(eigvals) > 1e-6
    eigvals = eigvals[valid_idx]
    eigvecs = eigvecs[:, valid_idx]
    nGotDim = len(eigvals)
    

    idx = 0
    while idx < nGotDim and np.abs(eigvals[idx] - 1) < 1e-12:
        idx += 1
    
    if idx > 1: 
        u = np.zeros((nSmp, idx))
        d_m = 1.0 / D_mhalf
        u[:, 0] = d_m / np.linalg.norm(d_m)

        for i in range(1, idx):
            u[:, i] = eigvecs[:, i]
            for j in range(i):
                u[:, i] -= np.dot(u[:, j], eigvecs[:, i]) * u[:, j]
            u[:, i] /= np.linalg.norm(u[:, i])
        
        eigvecs[:, :idx] = u

    Y = diags(D_mhalf).dot(eigvecs) if issparse(W) else np.diag(D_mhalf) @ eigvecs
    
    return Y[:, 1:], eigvals[1:]

def multi_cluster_fs(X, num_selected_features = -1, y = None, laplacian_reduction_dim = 5, W = None):
    
    """
    Perform multi-cluster feature selection via spectral clustering and LARS regression.
    This function selects a subset of features from the input data matrix X using a multi-cluster
    feature selection approach. In supervised mode (when labels `y` are provided), it generates a 
    class-based projection matrix. In unsupervised mode, it constructs a similarity graph (or uses a 
    user-provided affinity matrix `W`) and performs Laplacian Eigenmap embedding. Then, it uses the 
    LARS regression algorithm to select the most relevant features based on the learned embedding.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    num_selected_features : int, optional (default=-1)
        Number of features to select. If set to -1 or a value larger than the total number of features, 
        all features will be returned.
    y : array-like of shape (n_samples,), optional (default=None)
        Target labels for supervised feature selection. If None, unsupervised mode is used.
    laplacian_reduction_dim : int, optional (default=5)
        The dimensionality of the Laplacian Eigenmap embedding. If larger than the number of features, 
        it will be adjusted.
    W : array-like or sparse matrix, optional (default=None)
        Precomputed affinity/similarity matrix for unsupervised mode. If None, the affinity matrix will 
        be computed internally using a k-nearest neighbors graph.
    Returns
    -------
    feature_index : array-like of shape (num_selected_features,)
        Indices of the selected features, sorted in descending order of importance scores.
    Notes
    -----
    - In supervised mode, the function constructs a random projection for each class and 
        orthogonalizes it via QR decomposition.
    - In unsupervised mode, the similarity graph is computed over either the whole dataset or a subset 
        for efficiency.
    - Feature importance scores are aggregated from the absolute values of LARS regression coefficients 
        across all embedding dimensions.
    """
    
    n_samples, n_features = X.shape
    if(num_selected_features > n_features or num_selected_features < 0):
        num_selected_features = n_features
    if(laplacian_reduction_dim > n_features):
        laplacian_reduction_dim = min (n_features, 50)
    
    # 2. construct projection matrix Y
    if y is not None:  # supervised learning mode
        class_labels = np.unique(y)
        n_classes = len(class_labels)
        
        # generate random projection matrix Y
        np.random.seed(42)
        Y_rand = np.random.rand(n_classes, n_classes)
        Z = np.zeros((n_samples, n_classes))
        
        for i, label in enumerate(class_labels):
            idx = np.where(y == label)[0]
            Z[idx] = np.tile(Y_rand[i], (len(idx), 1))
        Q, R = qr(Z, mode='economic')
        Y = Q[:, 1:]  
        print(f'Y.shape = {Y.shape}')
        
    else:  # unsupervised learning mode
        if W == None:
            
            if n_samples > 3000:
                subset = np.random.choice(n_samples, 3000, replace=False)
                fea_subset = X[subset]
            else:
                fea_subset = X
            
            
            dists = np.sum(fea_subset ** 2, axis = 1)[:, np.newaxis] + \
                    np.sum(fea_subset ** 2, axis = 1) - \
                    2 * fea_subset.dot(fea_subset.T)
            median_dist = np.median(dists)
            W = kneighbors_graph(X, n_neighbors = 5, mode = 'distance', metric = 'euclidean')
            W.data = np.exp(-W.data ** 2 / median_dist)
            W = 0.5 * (W + W.T)  
        
        # laplacian eigenmap
        Y, eig_vals = _Eigenmap(W, laplacian_reduction_dim)
        print(f'eig_vals = {eig_vals}')
        Y = np.array(Y)
    
    # 3. feature selection using LARS
    lasso_models = []
    for i in range(Y.shape[1]):
        #model = Lars(n_nonzero_coefs = int(max(FeaNumCandi)*opts['ratio']), fit_intercept = False)
        #model_cv = LassoLarsCV(cv=5, max_iter=1000).fit(X, Y[:, i])
        #print("Optimal alpha:", model_cv.alpha_)
        model = Lars(n_nonzero_coefs = n_features, fit_intercept = False)
        #model = LassoLars(alpha = 1e-5, fit_intercept = True, copy_X = True)
        model.fit(X, Y[:, i])
        lasso_models.append(model)
    
    # 4. feature importance scores
    feature_scores = np.zeros(n_features)
    
    # 
    for model in lasso_models:
        coef_ = model.coef_
        #print(f'coef_ = {coef_}')
        #step = min(req_nonzero, coef_.shape[1] - 1)
        scores = np.abs(coef_)
        feature_scores += scores
    sorted_idx = np.argsort(feature_scores)[::-1]

    feature_index = sorted_idx
    return feature_index[0:num_selected_features]

def mici_fs(X, num_selected_feature, similarity_matrix = None):
    '''
    Selects features based on the Mutual Information and Cosine Similarity Index (MICI) method.
    This function selects a subset of features from the input data matrix `X` by maximizing the total pairwise similarity among the selected features, using a combination of variance and cosine similarity. The selection is performed via an iterative greedy search.
    Parameters:
        X : np.ndarray
            The input data matrix after feature extraction, of shape (n_samples, n_features).
        num_selected_feature : int
            The number of features to select.
        similarity_matrix : np.ndarray, optional
            Precomputed similarity matrix of shape (n_features, n_features). If None, cosine similarity is computed from `X`.
    Returns:
        selected_indices : np.ndarray
            Indices of the selected features, sorted in ascending order.
    Notes:
        - The function uses cosine similarity squared as the similarity measure.
        - The selection process is initialized randomly and refined iteratively to maximize the sum of pairwise similarities among selected features.
        - The higher the total pairwise similarity, the more important the selected features are considered.
    '''
    if(similarity_matrix is None):
        S_cos = cosine_similarity(X.T).astype(np.float64)
        S_cos [S_cos > 1] = 1
    else:
        S_cos = similarity_matrix
    S_cos = S_cos**2
    var_x = np.array(np.var(X, axis = 0)).astype(np.float64)
    n_samples, n_features = X.shape
    lambda_matrix = np.zeros((n_features, n_features), dtype = np.float64)
    for i in range(n_features): 
        for j in range(i):
            tmp = np.abs(var_x[i]+var_x[j])**2 - 4*var_x[i]*var_x[j]*(1-S_cos[i,j])
            lambda_matrix[i,j] = var_x[i] + var_x[j] - np.sqrt(tmp)
            lambda_matrix[j,i] = lambda_matrix[i,j]
    np.random.seed(42)
    selected_indices = np.sort(np.random.choice(n_features, size = num_selected_feature, replace = False))        
    iter_times = 10
    for iter in range(iter_times):
        err_old = lambda_matrix[np.ix_(selected_indices, selected_indices)].sum()
        err_new = err_old
        for i in range(num_selected_feature):
            for j in range(n_features):
                if (np.isin(j, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                err_tmp = lambda_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)].sum()
                if(err_tmp > err_new):
                    err_new = err_tmp
                    selected_indices[i] = j
        if(np.abs(err_new - err_old)/ np.abs(err_old + 1e-10) < 1e-6):
            break
    return selected_indices

def mutual_corre_fs(X, num_selected_feature, similarity_matrix = None):
    """
    Selects a subset of features based on mutual correlation using an iterative optimization approach.
    This function selects `num_selected_feature` features from the input data matrix `X` such that the sum of pairwise similarities (correlations) among the selected features is maximized. The selection is performed using a greedy iterative algorithm that swaps features to improve the total similarity score.
    Parameters
    ----------
    X : numpy.ndarray
        The input data matrix of shape (n_samples, n_features).
    num_selected_feature : int
        The number of features to select.
    similarity_matrix : numpy.ndarray, optional
        Precomputed feature similarity matrix of shape (n_features, n_features).
        If None, the cosine similarity of the transposed input matrix is used.
    Returns
    -------
    selected_indices : numpy.ndarray
        Array of indices of the selected features, sorted in ascending order.
    Notes
    -----
    - The function uses cosine similarity as the default similarity measure.
    - The selection process is stochastic due to random initialization (seeded for reproducibility).
    - The algorithm stops early if the improvement in the objective function is below a threshold.
    """
        
    if(similarity_matrix is None):
        similarity_matrix= np.abs(cosine_similarity(X.T))
        similarity_matrix [similarity_matrix > 1] = 1
    np.fill_diagonal(similarity_matrix, 0)

    n_samples, n_features = X.shape
    
    np.random.seed(42)
    selected_indices = np.sort(np.random.choice(n_features, size = num_selected_feature, replace = False))     
    iter_times = 10
    for iter in range(iter_times):
        err_old = similarity_matrix[np.ix_(selected_indices, selected_indices)].sum()
        err_new = err_old
        for i in range(num_selected_feature):
            for j in range(n_features):
                if (np.isin(j, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                err_tmp = similarity_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)].sum()
                if(err_tmp > err_new):
                    err_new = err_tmp
                    selected_indices[i] = j
        if(np.abs(err_new - err_old)/np.abs(err_old + 1e-10) < 1e-6):
            break
    return selected_indices

def mdcm_fs(X, num_selected_feature, covariance_matrix = None):
    """
    Performs feature selection using the Maximum Determinant Covariance Minimization (MDCM) algorithm.
    This method iteratively selects a subset of features that maximize the product of singular values (akin to the determinant)
    of the covariance submatrix aligning to the chosen features. The approach prefer subsets whose corresponding covariance 
    submatrix is well-conditioned in terms of spread of data.
    Args:
        X (numpy.ndarray): 2D data matrix of shape (n_samples, n_features).
        num_selected_feature (int): Number of features to select.
        covariance_matrix (numpy.ndarray, optional): Covariance matrix of shape (n_features, n_features). 
            If None, it will be computed from X.
    Returns:
        numpy.ndarray: Indices of selected features (length = num_selected_feature).
    """
    
    if(covariance_matrix is None):
       covariance_matrix = np.cov(X, rowvar = False)
    n_samples, n_features = X.shape
    np.random.seed(42)
    selected_indices = np.sort(np.random.choice(n_features, size = num_selected_feature, replace = False))        
    iter_times = 10
    for iter in range(iter_times):
        sub_conv_matrix = covariance_matrix[np.ix_(selected_indices, selected_indices)]
        _, s, _ = np.linalg.svd(sub_conv_matrix)
        err_old = math.prod(s)
        err_new = err_old
        for i in range(num_selected_feature):
            for j in range(n_features):
                if (np.isin(j, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                sub_conv_matrix = covariance_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)]
                _, s, _ = np.linalg.svd(sub_conv_matrix)
                err_tmp = math.prod(s)
                if(err_tmp > err_new):
                    err_new = err_tmp
                    selected_indices[i] = j
        if(np.abs(err_new - err_old)/np.abs(err_old + 1e-10) < 1e-6):
            break
    return selected_indices
    
def linear_dependency_fs(X, num_selected_feature, covariance_matrix = None):
    """
    Selects a subset of features from the input data matrix X by minimizing linear dependency among the selected features using an iterative SVD-based approach.

    Parameters:
        X (np.ndarray): Input data matrix of shape (n_samples, n_features).
        num_selected_feature (int): Number of features to select.
        covariance_matrix (np.ndarray, optional): Precomputed covariance matrix of shape (n_features, n_features). If None, it is computed from X.

    Returns:
        np.ndarray: Indices of the selected features (sorted array of length num_selected_feature).

    Notes:
        - The function uses a greedy iterative algorithm to maximize the sum of absolute values after projecting the selected features onto the right singular vectors of their covariance submatrix.
        - The selection process is repeated for a fixed number of iterations or until convergence.
        - Random seed is set to 42 for reproducibility.
    """

    if(covariance_matrix is None):
       covariance_matrix = np.cov(X, rowvar = False)
    n_samples, n_features = X.shape
    np.random.seed(42)
    selected_indices = np.sort(np.random.choice(n_features, size = num_selected_feature, replace = False))        
    iter_times = 10
    for iter in range(iter_times):
        sub_conv_matrix = covariance_matrix[np.ix_(selected_indices, selected_indices)]
        _, _, v = np.linalg.svd(sub_conv_matrix)
        E = X[:,selected_indices] @ v.T
        err_old = np.abs(E).sum()
        err_new = err_old
        for i in range(num_selected_feature):
            for j in range(n_features):
                if (np.isin(j, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                sub_conv_matrix = covariance_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)]
                _, _, v = np.linalg.svd(sub_conv_matrix)
                E = X[:,selected_indices_tmp] @ v.T
                err_tmp = np.abs(E).sum()
                if(err_tmp > err_new):
                    err_new = err_tmp
                    selected_indices[i] = j
        if(np.abs(err_new - err_old)/np.abs(err_old + 1e-10) < 1e-6):
            break
    return selected_indices
def mpmr_fs(X, num_selected_feature = -1, correlatin_matrix = None):
    """
    Performs Minimum Projection Minimum Redundancy Feature Selection (MPMR-FS) for unsupervised feature subset selection.
    Args:
        X (numpy.ndarray): The data matrix of shape (n_samples, n_features).
        num_selected_feature (int, optional): Number of features to select. If -1, all features are selected. Default is -1.
        correlatin_matrix (numpy.ndarray, optional): Precomputed feature correlation/affinity matrix. If None, it is computed as X.T @ X.
    Returns:
        numpy.ndarray: Sorted indices of the selected features of length `num_selected_feature`.
    Description:
        The mpmr_fs function iteratively refines a subset of features based on their reconstruction error,
        using an iterative projection approach. It alternates feature swapping to minimize reconstruction loss subject to
        mutual feature correlations, stopping early if the improvement is below a threshold.
    Notes:
        - Requires numpy.
        - Set random seed for reproducibility.
        - Early stops if relative improvement is sufficiently small.
    """
    
    if(correlatin_matrix is None):
        correlatin_matrix = X.T @ X
    n_samples, n_features = X.shape
    np.random.seed(42)
    selected_indices = np.sort(np.random.choice(n_features, size = num_selected_feature, replace = False))        
    iter_times = 10
    for iter in range(iter_times):
        sub_corr_matrix = correlatin_matrix[np.ix_(selected_indices, selected_indices)]
        E = X - X[:, selected_indices] @ np.linalg.pinv(sub_corr_matrix) @ correlatin_matrix[selected_indices,:]
        err_old = np.linalg.norm(E)
        err_new = err_old
        for i in range(num_selected_feature):
            for j in range(n_features):
                if (np.isin(j, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                sub_corr_matrix = correlatin_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)]
                E = X - X[:, selected_indices] @ np.linalg.pinv(sub_corr_matrix) @ correlatin_matrix[selected_indices,:]
                err_tmp = np.linalg.norm(E)
                if(err_tmp < err_new):
                    err_new = err_tmp
                    selected_indices[i] = j
        if(np.abs(err_new - err_old)/np.abs(err_old + 1e-10) < 1e-6):
            break
    return selected_indices


def _Go(A, m, k):
    """
    Python version of GO, called by RSPCA
    paprameters:
        A : numpy.ndarray - d×d matrix usually covariance matrix
        m : int - dimensions after projection
        k : int - number of features selected in each iteration
    returns:
        W : numpy.ndarray -  projection matrix after optimization (d×m)
    """
    d = A.shape[0] 
    Aopt = A.copy()  

    # check if rank of A is greater than m
    if np.linalg.matrix_rank(A) > m:
        eigenvalues, eigenvectors = eigs(A, k=m, which='LM')
        
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # sort the eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
   
        A = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # choose the k largest diagonal elements
    diagA = np.diag(A)
    ind = np.argsort(-diagA)  
    opt_index = np.sort(ind[:k]) 

    # initialization of projection matrix
    W = np.zeros((d, m))

    # get the submatrix and compute eigenvectors
    Aopt_sub = Aopt[np.ix_(opt_index, opt_index)]
    eigenvalues_sub, eigenvectors_sub = eigh(Aopt_sub, 
                                           subset_by_index=(Aopt_sub.shape[0]-m, 
                                                           Aopt_sub.shape[0]-1))
    
    W[opt_index, :] = eigenvectors_sub

    return W

def _IPU(A, d, m, k, W0 = None):
    """
    Python version of IPU, called by RSPCA
    paprameters:
        A : numpy.ndarray - d×d matrix usually covariance matrix
        d : int - number of features
        m : int - dimensions after projection
        k : int - number of features selected in each iteration
        W0 : numpy.ndarray -  initial projection matrix(d×m), default is None
    returns:
        W : numpy.ndarray -  projection matrix after optimization (d×m)
        opt_index : numpy.ndarray - index of selected features
    """
    NITER = 100  # iterations
    
    # =1=  initialization of W0
    if W0 is None:
        W0 = _Go(A, m, k)
    W = W0.copy()
    
    # =2= check rank of A
    if np.linalg.matrix_rank(A) <= m:
        WtA = W.T @ A         # W'*A
        inv_term = pinv(WtA @ W)  # pinv(W'*A*W)
        P = A @ W @ inv_term @ WtA
        # =3.2= select top-k indices
        diagP = np.diag(P)
        # sort the indices and take top-k
        opt_index = np.argsort(-diagP)[:k]
        return W, opt_index 
       # =3= main iteration loop
    V = np.zeros((k, m),dtype= float)
    for iter in range(NITER):
        # =3.1=  compute projection matrix P
        WtA = W.T @ A         # W'*A
        inv_term = pinv(WtA @ W)  # pinv(W'*A*W)
        P = A @ W @ inv_term @ WtA
        
        # =3.2=  choose top-k indices
        diagP = np.diag(P)
        #sort the indices and take top-k
        opt_index = np.argsort(-diagP)[:k]  
    
        
        # =3.3=  eigenvalue decomposition of submatrix
        Aopt = A[np.ix_(opt_index, opt_index)]  # 子矩阵提取
        eig_vals, eig_vecs = eigh(Aopt, subset_by_index=(Aopt.shape[0]-m, Aopt.shape[0]-1))
        V_old = V
        V = eig_vecs[:, -m:]  # take the first m eigenvectors
        
        # =3.4= update projection matrix 
        W.fill(0)  
        W[opt_index, :] = V  
        if(np.linalg.norm(V-V_old)/np.linalg.norm(V) < 0.001):
            break
    return W, opt_index


def rs_pca(X, sigma, num_selected_feature):
    """
    python implementation of RSPCA algorithm
    parameters:
        X : numpy.ndarray - input data (dim x num)
        sigma : float - regularization parameter(0.001-1000)
        num_selected_featuure : int - the number of selected features
    returns:
        P : numpy.ndarray - projectrion matrix (dim x pdim)
        opt_index : numpy.ndarray - index of selected features
    """
    iter_times = 100
    n_samples, n_feautres = X.shape  # size of X
    projection_dim = num_selected_feature
    #random initialization of orthogonal projection matrix
    np.random.seed(42)
    P, _ = np.linalg.qr(np.random.rand(n_samples, projection_dim))  # produce the othogonal basis by QR decomposition
    
    # initialization of vectors
    d = np.ones((n_feautres, 1))       # weight vector (num x 1)
    e = np.ones((n_feautres, 1))       # one vector (num x 1) 
    obj_new = 0
    # main iteration loop (original MATLAB code fixed 1 iteration)
    for iter in range(iter_times):
        obj_old = obj_new
        # =3.1= compute weighted mean vector b
        bi = X @ d              # X*d (n_samples x 1)
        b = bi / np.sum(d)      # normalization (n_samples x 1)
        
        # =3.2= compute centered data matrix A
        A = X - b @ e.T         # X - b*ones(1,n_feautres) (n_samples x n_feautres)
        
        # =3.3=  compute residual matrix B
        B = A - P @ (P.T @ A)   # A - P*(P'*A) (n_samples x n_feautres)
        
        # =3.4=  compute residual norm and update weight d
        Bi = np.sqrt(np.sum(B**2, axis = 0)) + np.finfo(float).eps  # L2 norm of each column (1 x n_feautres)
        d = (1 + sigma) * ((Bi + 2 * sigma) / (2 * (Bi + sigma)**2))  # weight update formula d = (1+sigma)*((Bi+2*sigma)./(2*(Bi+sigma).^2)) 
        d = d.reshape(-1, 1)    # reshape to column vector(n_feautres x 1) 
        
        # =3.5= construct the centering matrix Hd (MATLAB  Hd = D - (D*e*e'*D)/(e'*D*e)) 
        D = np.diag(d.flatten()) # diag matrix (n_feautres x n_feautres)
        temp = D @ e             # temp variable (n_feautres x 1) 
        denom = e.T @ temp       # scalar denominator
        Hd = D - (temp @ temp.T) / denom  # centering matrix (n_feautres x n_feautres) 
        
        # =3.6= compute covariance matrix Q and call IPU algorithm
        Q = X @ Hd @ X.T         # covariance matrix (dim x dim) weight 
        
        P, opt_index = _IPU(Q, n_samples, projection_dim, num_selected_feature, P)  # call IPU to optimize projection matrix  m<=k<=d, d, m, k, 
        
        Oi = ((1 + sigma)*(Bi**2))/(Bi + sigma)#
        obj_new = sum(Oi) #
        if(np.abs(obj_new - obj_old)/(obj_old + 1e-10) < 0.001):
            break
        
    return P, opt_index, obj_new