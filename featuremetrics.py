# featuremetrics.py
'''
A collection of feature selection function and clustering evaluation metrics for machine learning, 
including Laplacian Score, similarity entropy, SVD entropy, variance-based scores, 
mutual correlation, maximum information compression index, covariance determinant, 
linear representation error, ERFS error, RSPCA, MCFS, and various clustering evaluation utilities.
Functions:
- laplacian_score(X, W=None): 
- similarity_entropy(X, similarity_matrix=None): 
- svd_entropy(X, convariance_matrix=None): 
- variance_score(X): 
- mutual_corre(X, selected_cols, similarity_matrix=None): 
- max_info_compress_index(X, selected_cols, similarity_matrix=None): 
- max_determinant_of_covmatrix_and_linear_dependency(X, selected_cols, convariance_matrix=None): 
- mrmr_value(selected_cols, F_values, corr_matrix): 
    Compute the mRMR (minimum Redundancy Maximum Relevance) value for selected features.
- linear_rep_error(X, selected_cols): 
- erfs_error(X, y, selected_cols, correlation_matrix=None): 
- _Go(A, m, k): 
    Internal function for projection matrix optimization.
- _IPU_evluate(A, d, m, k, selected_incices, W0=None): 
    Internal function for iterative projection update.
- _rs_pca_evaluate(X, sigma, selected_cols): 
    Internal function for RSPCA evaluation metric.
- rspca_value(X, selected_cols): 
    Compute RSPCA value for selected features.
- mcfs_parameter(X, k_clusters=5): 
- mcfs_score(X, yks, x_corr, xty, selected_cols): 
    Calculate the MCFS regression error for selected features.
- get_cluster_label_distribution(true_labels, pred_clusters): 
- find_optimal_cluster_label_mapping(cluster_label_dist): 
    Find optimal cluster-to-label mapping using the Hungarian algorithm.
- multi_label_clustering_accuracy(true_labels, pred_labels, average='weighted'): 
- evaluate_clustering(true_labels, pred_clusters): 
    Evaluate multi-label clustering accuracy (mapping clusters to labels).
- evaluate_clustering(X, labels, true_labels): 
    Evaluate clustering performance using silhouette score and adjusted Rand index.
- assign_cluster_labels(true_labels, cluster_labels, top_k=3): 
    Assign high-frequency label sets to each cluster.
- partial_match_accuracy(y_true, y_pred): 
    Compute partial match accuracy for multi-label predictions.
- cluster_acc(y_true, y_pred): 
    Compute clustering accuracy with optimal label matching.
Dependencies:
    numpy, pandas, scipy, scikit-learn, skfeature
References:
    - He, X., Cai, D., & Niyogi, P. (2005). "Laplacian Score for Feature Selection". NIPS.
    - MCFS, RSPCA, mRMR, and other feature selection literature.
'''
import numpy as np
import pandas as pd
import math, os,time,copy
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import pinv, eigh
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from skfeature.utility import construct_W
from sklearn.metrics import f1_score, silhouette_score, adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment


def laplacian_score(X, W = None):
    """
    Compute the Laplacian Score for feature selection.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    W : ndarray or sparse matrix, shape (n_samples, n_samples)
        Affinity matrix capturing data similarity.
    
    Returns:
    Y : ndarray, shape (n_features,) (1-LaplacianScore) 
        Laplacian Score for each feature. Higher scores indicate more important features.
    
    Reference:
    He, X., Cai, D., & Niyogi, P. (2005). 
    "Laplacian Score for Feature Selection". NIPS.
    """
    if W == None :
        X_c = copy.copy(X)
        W = construct_W.construct_W(X_c)
    
    D = diags(np.array(W.sum(axis=1)).flatten(), format='csr')
    L = D - W
    
    # compute score for each feature
    scores = []
    for i in range(X.shape[1]):
        fr = X[:, i]
        numerator = fr.T @ L @ fr  # mumerator: fr^T L fr
        denominator = fr.T @ D @ fr + 1e-8  # denominator: fr^T D fr
        scores.append(numerator / denominator)
    
    return np.array(scores)

def similarity_entropy(X, similarity_matrix = None):
    """
    Compute the Similarity Entropy score for feature selection.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    similarity_matrix : ndarray, shape (n_features, n_features), optional
        Precomputed feature similarity matrix. If None, cosine similarity will be used.
    
    Returns:
    se : ndarray, shape (n_features,)
        Similarity Entropy score for each feature. Higher scores indicate more important features.
    """
    # Compute similarity matrix if not provided
    if(similarity_matrix is None):
        similarity_matrix = np.abs(cosine_similarity(X.T))
        similarity_matrix [similarity_matrix > 1] = 1
    E = -(similarity_matrix*np.log2(similarity_matrix + 1e-9) +
          (1 - similarity_matrix)*np.log2(1 - similarity_matrix + 1e-9))
    se = []
    for i in range(similarity_matrix.shape[1]):
        se_i = 2*np.sum(E[i,:]) - E[i,i]
        se.append(se_i)
    return np.array(se)

def _svd_val_entropy(Cov_matrix):
    _, ev, _ = np.linalg.svd(Cov_matrix)
    #ev = np.real(eigvals(Cov_matrix))
    ev = ev/ev.sum() + 1e-10
    se = -ev*np.log2(ev)
    return se.sum()

def svd_entropy(X, convariance_matrix = None):
    """
    Compute SVD-based entropy for feature selection.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    covariance_matrix : ndarray, shape (n_features, n_features), optional
        Precomputed covariance matrix. If None, will be computed from X.
    
    Returns:
    se : ndarray, shape (n_features,)
        SVD entropy score for each feature. Higher scores indicate more important features.
    """
    # Compute covariance matrix if not provided
    if(convariance_matrix is None):
        Cov_matrix = np.cov(X, rowvar=False)
    else:
        Cov_matrix = convariance_matrix
    se_F = _svd_val_entropy(Cov_matrix)
    se = []
    ind = range(Cov_matrix.shape[0])
    for i in ind:
        sub_ind = np.delete(ind,i)
        sub_matrix = Cov_matrix[np.ix_(sub_ind, sub_ind)]
        se_i = se_F - _svd_val_entropy(sub_matrix)
        se.append(se_i)
    return np.array(se)

def variance_score(X):
    """
    Compute variance-based feature importance scores.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    
    Returns:
    variances : ndarray, shape (n_features,)
        Variance scores for each feature. Higher scores indicate more important features.
    """
    return np.var(X,axis=0)

def mutual_corre(X, selected_cols, similarity_matrix = None):
    """
    Compute mutual correlation score for selected features.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    selected_cols : ndarray, shape (k_selected_features,)
        Indices of selected features.
    similarity_matrix : ndarray, shape (n_features, n_features), optional
        Precomputed feature similarity matrix. If None, cosine similarity will be used.
    
    Returns:
    score : float
        Mutual correlation score for the selected feature subset.
    """
    # Compute similarity matrix if not provided
    if(similarity_matrix is None):
        X_selected = X[:, selected_cols]
        sm = np.abs(cosine_similarity(X_selected.T))
        sm [similarity_matrix > 1] = 1
    else:
        sm = similarity_matrix[np.ix_(selected_cols, selected_cols)]
    return (sm.sum() - np.trace(sm))/2

def max_info_compress_index(X, selected_cols, similarity_matrix = None):
    """
    Compute Maximum Information Compression Index (MICI) for selected features.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    selected_cols : ndarray, shape (k_selected_features,)
        Indices of selected features.
    similarity_matrix : ndarray, shape (n_features, n_features), optional
        Precomputed feature similarity matrix. If None, cosine similarity will be used.
    
    Returns:
    mic_score : float
        MICI score for the selected feature subset.
    """
    X_selected = X[:, selected_cols]
    if(similarity_matrix is None):
        S_cos = cosine_similarity(X_selected.T).astype(np.float64)
        S_cos [S_cos > 1] = 1
    else:
        S_cos = similarity_matrix[np.ix_(selected_cols, selected_cols)]
    S_cos = S_cos**2
    var_x = np.array(np.var(X_selected, axis = 0)).astype(np.float64)
    n = X_selected.shape[1]
    lambda_matrix = np.zeros((n, n), dtype = np.float64)
    for i in range(n): 
        for j in range(i):
            tmp = np.abs(var_x[i]+var_x[j])**2 - 4*var_x[i]*var_x[j]*(1-S_cos[i,j])
            lambda_matrix[i,j] = var_x[i] + var_x[j] - np.sqrt(tmp + 1e-8)
    return lambda_matrix.sum()

def max_determinant_of_covmatrix_and_linear_dependency(X, selected_cols, convariance_matrix = None):
    """
    Compute covariance matrix determinant and linear dependency for selected features.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    selected_cols : ndarray, shape (k_selected_features,)
        Indices of selected features.
    covariance_matrix : ndarray, shape (n_features, n_features), optional
        Precomputed covariance matrix. If None, will be computed from X.
    
    Returns:
    det2 : float
        Determinant of the covariance matrix for selected features.
    ld_score : float
        Linear dependency score for selected features.
    """
    X_selected = X[:, selected_cols]
    if(convariance_matrix is None):
        Cov_matrix = np.cov(X_selected, rowvar = False)
    else:
        Cov_matrix = convariance_matrix[np.ix_(selected_cols, selected_cols)]
    #ev = np.real(eigvals(Cov_matrix))
    _, s, v = np.linalg.svd(Cov_matrix)
    #E = X_selected @ v.T
    #ld = np.linalg.norm(E, ord ='fro')
    det = math.prod(s)
    E = X_selected @ v.T
    return det, np.abs(E).sum()

def mrmr_value(selected_cols, F_values, corr_matrix):
    n = selected_cols.shape[0]
    #corr_matrix = np.interp(corr_matrix, (0, 1), (0.3, 1))
    #sum_corr = (corr_matrix.sum() - n) / 2
    mrmr_v = 0
    for i in range(n):
        corr = corr_matrix[selected_cols[i],selected_cols].sum() - 1
        #for j in range(n):
        #    if(j == i):
        #        continue
        #    else:
         #       corr += corr_matrix[selected_cols[i],selected_cols[j]]
        mrmr_v += (F_values[selected_cols[i]]) / corr
    return mrmr_v

def linear_rep_error(X, selected_cols):
    """
    Compute linear representation error for selected features.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    selected_cols : ndarray, shape (k_selected_features,)
        Indices of selected features.
    
    Returns:
    error : float
        Frobenius norm of the representation error.
    """
    X_s = X[:, selected_cols]
    # solve least squares problem
    B, residuals, _, _ = np.linalg.lstsq(X_s, X, rcond=None)
    #  compute residuals
    R = X - X_s @ B
    error = np.linalg.norm(R, 'fro')  # Frobenius norm
    return error

def erfs_error(X, y, selected_cols, correlation_matrix = None):
    """
    Compute Error-Reduction Feature Selection (ERFS) error.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    y : ndarray, shape (n_samples,)
        Target vector.
    selected_cols : ndarray, shape (k_selected_features,)
        Indices of selected features.
    correlation_matrix : ndarray, shape (n_features, n_features), optional
        Precomputed correlation matrix. If None, will be computed from X.
    
    Returns:
    error : float
        L2 norm of the prediction error.
    """
    X_s = X[:, selected_cols]
    #X_s = X_s - np.mean(X_s, axis = 0)
    if(correlation_matrix is None):
        Cor_matrix =  X_s.T @ X_s
    else:
        Cor_matrix = correlation_matrix[np.ix_(selected_cols, selected_cols)]
    w = np.linalg.pinv(Cor_matrix) @ X_s.T @ y
    R = y - X_s @ w  
    error = np.linalg.norm(R, ord = 2)  # Frobenius norm
    return error

def _Go(A, m, k):
    """
    Python version of GO
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

def _IPU_evluate(A, d, m, k, selected_incices, W0 = None):
    """
    Python version of IPU
    paprameters:
        A : numpy.ndarray - d×d matrix usually covariance matrix
        d : int - number of features
        m : int - dimensions after projection
        k : int - number of features selected in each iteration
        selected_incices:  selected feature indices
        W0 : numpy.ndarray -  initial projection matrix(d×m), default is None
    returns:
        W : numpy.ndarray -  projection matrix after optimization (d×m)
    """
    NITER = 1 # iter
    # =1= initialization of W0
    if W0 is None:
        W0 = _Go(A, m, k)
    W = W0.copy()
    
    # =2= check rank of A
    if np.linalg.matrix_rank(A) <= m:
        return W  # 
    # =3= main iteration loop
    V = np.zeros((k, m),dtype = float)
    for iter in range(NITER):
                
        # =3.1= eigen value decomposition of submatrix
        Aopt = A[np.ix_(selected_incices, selected_incices)]  # 子矩阵提取
        eig_vals, eig_vecs = eigh(Aopt, subset_by_index=(Aopt.shape[0]-m, Aopt.shape[0]-1))
        V_old = V
        V = eig_vecs[:, -m:]  # 取前m个最大特征值对应的特征向量
        
        # =3.2= update projection matrix 
        W.fill(0) 
        W[selected_incices, :] = V  
        if(np.linalg.norm(V - V_old )/np.linalg.norm(V) < 0.0001):
            break
    return W

def _rs_pca_evaluate(X, sigma, selected_cols)->float:
    """
    python implementation of RSPCA evaluation metric algorithm
    parameters:
        X : numpy.ndarray - input data (dim x num)
        sigma : float - regularization parameter (0.001-1000)
        seleted_clos : numpy.ndarray - selected features
    returns:
        obj_val : numpy.ndarray - metric value of feature subsets
    """
    iter_times = 100
    n_samples, n_feautres = X.shape  
    num_selected_feature = selected_cols.shape[0]
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
        
        P = _IPU_evluate(Q, n_samples, projection_dim, num_selected_feature, selected_incices = selected_cols, W0 = P)  # call IPU to optimize projection matrix  m<=k<=d, d, m, k, 
        Oi = ((1 + sigma)*(Bi**2))/(Bi + sigma)#
        obj_new = sum(Oi) #
        if(np.abs(obj_new - obj_old)/(obj_old + 1e-10) < 0.001):
            break        
    return obj_new



def rspca_value(X, selected_cols):
    sigma = 0.1
    rsp_v = _rs_pca_evaluate(X = X.T,sigma = sigma, selected_cols = selected_cols)
    return rsp_v

def mcfs_parameter(X, k_clusters = 5):
    """
    Precompute parameters for MCFS.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Data matrix where rows are samples, columns are features.
    k_clusters : int, optional
        Number of clusters for Laplacian eigenmap computation.
    
    Returns:
    yks : ndarray, shape (n_samples, k_clusters)
        Laplacian matrix eigenvectors.
    x_corr : ndarray, shape (n_features, n_features)
        Covariance matrix.
    xty : ndarray, shape (n_features, k_clusters)
        Product of X transpose and yks.
    """
    X_c = copy.copy(X)
    W = csr_matrix(construct_W.construct_W(X_c, neighbor_mode = 'knn', weight_mode= 'binary', k = 5))
    W_sum = np.array(np.abs(W.sum(axis = 1)))
    
    D = diags(W_sum.flatten(), format = 'csr')
    L = D - W
    sigma = 0.01
    eigen_vals, yks = eigsh(
        A = L, 
        k = k_clusters, 
        M = D, 
        which ='LM',  # Largest Magnitude
        sigma = sigma,
        tol = 1e-6, 
        maxiter = 10000
    )
    #print(f'eigen_vals = {eigen_vals}')
    x_corr = np.cov(X, rowvar = False)
    xty = X.T @ yks
    return yks, x_corr, xty
    
def mcfs_score(X, yks, x_corr,xty,selected_cols):
    '''
    Calculates the MCFS regression error ||yk - Xak|| for selected features.
    Parameters:
        X (np.ndarray): The original feature matrix of shape (n_samples, n_features).
        yks (np.ndarray): Precomputed Laplacian matrix eigenvectors of shape (n_samples, k_clusters).
        x_corr (np.ndarray): Precomputed inverse covariance matrix of shape (n_features, n_features).
        xty (np.ndarray): Precomputed X' * yk matrix of shape (n_features, k_clusters).
        selected_cols (list or np.ndarray): Indices of selected feature columns.
    Returns:
        mc_score (float): The Frobenius norm of the regression error ||yk - Xak||.
    '''
    x_corr_inv = np.linalg.pinv(x_corr[np.ix_(selected_cols, selected_cols)])
    xty_selected = xty[selected_cols,:]# n_seletedfeature * k_clusters
    a = x_corr_inv @ xty_selected   # n_seletedfeature * k_clusters
    err = yks - X[:,selected_cols] @ a
    mc_score = np.linalg.norm(err, 'fro') 
    return mc_score


def get_cluster_label_distribution(true_labels, pred_clusters):
    """
    Compute label distribution within each cluster.
    
    Parameters:
    true_labels : ndarray, shape (n_samples, n_labels)
        True multi-label matrix where 1 indicates label presence.
    pred_clusters : ndarray, shape (n_samples,)
        Predicted cluster assignments for each sample.
    
    Returns:
    cluster_label_dist : dict
        Dictionary mapping cluster indices to label count arrays.
    """
    n_clusters = len(np.unique(pred_clusters))
    n_labels = true_labels.shape[1]
    
    cluster_label_dist = {c: np.zeros(n_labels) for c in range(n_clusters)}
    
    for i, cluster in enumerate(pred_clusters):
        for label in np.where(true_labels[i] == 1)[0]:
            cluster_label_dist[cluster][label] += 1
    
    return cluster_label_dist

def find_optimal_cluster_label_mapping(cluster_label_dist):
    """
    Find optimal cluster-to-label mapping using Hungarian algorithm.
    
    Parameters:
    cluster_label_dist : dict
        Dictionary of label distributions per cluster.
    
    Returns:
    cluster_to_label : dict
        Optimal mapping from cluster indices to label indices.
    """
    clusters = sorted(cluster_label_dist.keys())
    n_clusters = len(clusters)
    n_labels = len(cluster_label_dist[clusters[0]])
    
    # constructing a cost matrix
    cost_matrix = np.zeros((n_clusters, n_labels))
    for i, cluster in enumerate(clusters):
        for j in range(n_labels):
            cost_matrix[i, j] = -cluster_label_dist[cluster][j] 
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create mapping dictionary
    cluster_to_label = {}
    for cluster_idx, label_idx in zip(row_indices, col_indices):
        cluster_to_label[clusters[cluster_idx]] = label_idx
    
    return cluster_to_label

def multi_label_clustering_accuracy(true_labels, pred_labels, average='weighted'):
    """
    Evaluate multi-label clustering accuracy.
    
    Parameters:
    true_labels : ndarray, shape (n_samples, n_labels)
        True multi-label matrix.
    pred_labels : ndarray, shape (n_samples, n_labels)
        Predicted multi-label matrix.
    average : str, optional
        F1-score averaging method ('micro', 'macro', 'weighted').
    
    Returns:
    accuracy : float
        Proportion of samples with at least one correct label match.
    f1 : float
        Multi-label F1 score.
    detailed_results : list of dict
        Per-sample evaluation details.
    """
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    if not isinstance(pred_labels, np.ndarray):
        pred_labels = np.array(pred_labels)
    
    n_samples = true_labels.shape[0]
    correct = 0
    detailed_results = []
    
    for i in range(n_samples):
        true = set(np.where(true_labels[i] == 1)[0])
        pred = set(np.where(pred_labels[i] == 1)[0])
        
        # check has intersection or not 
        if true & pred:
            correct += 1
            match = True
        else:
            match = False
        
        detailed_results.append({
            'sample_idx': i,
            'true_labels': list(true),
            'pred_labels': list(pred),
            'is_correct': match
        })
    
    accuracy = correct / n_samples
    f1 = f1_score(true_labels, pred_labels, average=average)
    
    return accuracy, f1, detailed_results

def evaluate_clustering(true_labels, pred_clusters):
    """
    Evaluate multi-label clustering accuracy.
    
    Parameters:
    true_labels : ndarray, shape (n_samples, n_labels)
        True multi-label matrix.
    pred_labels : ndarray, shape (n_samples, )
        Predicted single-label matrix.
    average : str, optional
        F1-score averaging method ('micro', 'macro', 'weighted').
    
    Returns:
    accuracy : float
        Proportion of samples with at least one correct label match.
    f1 : float
        Multi-label F1 score.
    detailed_results : list of dict
        Per-sample evaluation details.
    """
    # compute cluster-label distribution
    n_labels = true_labels.shape[1]
    cluster_label_dist = get_cluster_label_distribution(true_labels, pred_clusters)
    #print(f'clusterlbldist = {cluster_label_dist}')
    # find_optimal_cluster_label_mapping
    cluster_to_label = find_optimal_cluster_label_mapping(cluster_label_dist)
    #print(f'cluster_to_lbl = {cluster_to_label}')
    # transform the predicted clusters to multi-label format
    pred_labels = np.zeros_like(true_labels)
    for i, cluster in enumerate(pred_clusters):
        if cluster in cluster_to_label:
            pred_label = cluster_to_label[cluster]
            pred_labels[i, pred_label] = 1
    accuracy, f1, _ = multi_label_clustering_accuracy(true_labels, pred_labels)
    return accuracy, f1, cluster_to_label


def cluster_acc(y_true, y_pred):
    """
    计算聚类准确率（带最优标签匹配）
    :param y_true: 真实标签 (n_samples,)
    :param y_pred: 聚类标签 (n_samples,)
    :return: ACC ∈ [0, 1]
    """
    # 构建混淆矩阵
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
    # 匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # 计算正确匹配数
    correct_matches = cm[row_ind, col_ind].sum()
    acc = correct_matches / len(y_true)
    return acc