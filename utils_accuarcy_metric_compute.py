'''
This module provides utility functions for evaluating feature selection methods and computing accuracy and metric scores
for classification, regression, and clustering tasks. It supports random feature selection, metric computation for feature
combinations, and evaluation of various feature selection algorithms using multiple machine learning models.
Functions
---------
random_feature_selection_classification(X_train, X_test, y_train, y_test, num_sets, num_total_features, models, result_path, new_iter_times=50, max_iter_times=500)
    Randomly selects feature subsets for classification, evaluates model accuracy, and saves results.
compute_metrics_for_combinations_for_classification(X_metric_use, y_metric_use, X_org, num_sets, num_total_features, result_path, metric_path, is_overwrite_metric=True)
    Computes a variety of feature metrics for all combinations of selected features in classification tasks.
fs_methods_evaluation_for_classification(X_fsmethod_use, X_test, X_org, y_fsmethod_use, y_test, num_sets, num_total_features, models, result_path)
    Evaluates multiple feature selection methods for classification and computes accuracy metrics for various models.
random_feature_selection_regression(X_train, X_test, y_train, y_test, num_sets, num_total_features, models, result_path, new_iter_times=50, max_iter_times=500)
    Randomly selects feature subsets for regression, evaluates model performance, and saves results.
compute_metrics_for_combinations_for_regression(X_metric_use, X_org, y_metric_use, num_sets, num_total_features, result_path, metric_path, is_overwrite_metric=True)
    Computes feature metrics for all combinations of selected features in regression tasks.
fs_methods_evaluation_for_regression(X_fsmethod_use, X_test, X_org, y_fsmethod_use, y_test, num_sets, num_total_features, models, result_path)
    Evaluates multiple feature selection methods for regression and computes performance metrics for various models.
random_feature_selection_clustering(X, y_true, num_sets, num_total_features, models, result_path, is_multiple_labels=False, multi_lbl_true=None, new_iter_times=50, max_iter_times=500)
    Randomly selects feature subsets for clustering, evaluates clustering accuracy, and saves results.
compute_metrics_for_combinations_for_clustering(X, X_org, num_sets, num_total_features, num_clusters, result_path, metric_path, is_overwrite_metric=True)
    Computes feature metrics for all combinations of selected features in clustering tasks.
fs_methods_evaluation_for_clustering(X, X_org, y_true, num_sets, num_total_features, models, num_clusters, result_path, is_multiple_labels=False, multi_lbl_true=None)
    Evaluates multiple feature selection methods for clustering and computes clustering accuracy metrics for various models.
- The module supports a wide range of feature selection metrics, including variance, entropy-based, Laplacian score, SPECtrum,
  mutual correlation, MICI, MDCM, linear dependency, MPMR, MCFS, RSPCA, ANOVA F, ReliefF, Info Gain, MRMR, and ERFS.
- Results are saved as CSV files for each feature set size and method.
- The module is designed to handle both single-label and multi-label clustering scenarios.
'''
import numpy as np
import pandas as pd
import os,sys,math
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif, mutual_info_classif,f_regression,mutual_info_regression
from skfeature.function.similarity_based import SPEC
from mrmr import mrmr_classif, mrmr_regression
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from skrebate import ReliefF
from sklearn.neighbors import NearestNeighbors
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from featuremetrics import *
from featureselectionmethod import *
from custom_processor import *



def random_feature_selection_classification(X_train, X_test, y_train, y_test, 
                                            num_sets, num_total_features, models, 
                                            result_path, new_iter_times = 50, max_iter_times = 500):
    """
    Performs random feature selection for classification tasks and evaluates model accuracy.
    For each specified number of features to select, this function randomly samples unique feature subsets,
    trains and evaluates the provided classification models on these subsets, and saves the results.
    It avoids recomputation by checking previously saved results.
    Args:
        X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        X_test (np.ndarray): Test feature matrix of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        num_sets (list or iterable): List of integers specifying the number of features to select in each experiment.
        num_total_features (int): Total number of available features.
        models (dict): Dictionary mapping model names to sklearn-like classifier instances.
        result_path (str): Base path for saving result CSV files.
        new_iter_times (int, optional): Number of new random feature combinations to try per setting. Default is 50.
        max_iter_times (int, optional): Maximum number of total iterations (feature combinations) per setting. Default is 500.
    Notes:
        - If the number of features to select equals the total number of features, only one iteration is performed.
        - If the number of possible combinations is less than `max_iter_times`, all combinations are tried.
        - Previously computed feature combinations are loaded from CSV files to avoid duplication.
        - Results are saved incrementally to CSV files named according to the number of features selected.
    Returns:
        None. Results are saved to disk.
    """
    
    for num_select in num_sets:
        _iter_times = new_iter_times# real loop times
        max_iter_times_curr = max_iter_times
        num_columns_to_select = num_select
        print(f'\nprocessing choosing {num_select} features...', end =' ', flush = True)
        if num_columns_to_select == num_total_features:
            max_iter_times = 1
            _iter_times = 1
        if num_columns_to_select < 5 or num_total_features - num_columns_to_select < 5 :
            max_iter_times_curr = min(math.comb(num_total_features, num_select), max_iter_times)
        ########### Read saved results, check if the current feature combination has been computed ##########
        csv_result_file = f'{result_path}_classification_{num_select}.csv'
        if os.path.exists(csv_result_file):
            existed_featurecombinations = get_featureconbinations_from_csv(csv_result_file, num_columns_to_select, num_accurices_cols = 5)
            if(existed_featurecombinations.shape[0] > max_iter_times):
                continue
            _iter_times = min(new_iter_times, max_iter_times_curr - existed_featurecombinations.shape[0])
        else:
            _iter_times = min(new_iter_times, max_iter_times_curr)
            existed_featurecombinations = np.zeros((1,num_columns_to_select),dtype=int)
            np.expand_dims(existed_featurecombinations, axis = 0)

        for i in range(_iter_times):
            print(i + 1, end =' ',flush = True)
            random_column_indices = np.sort(np.random.choice(num_total_features, size = num_columns_to_select, replace = False))
            if(existed_featurecombinations.shape[0] > 1):# check if the random feature combination already exists
                while check_featurecombination_exist(existed_featurecombinations, random_column_indices):
                    random_column_indices = np.sort(np.random.choice(num_total_features, size = num_columns_to_select, replace = False))
            if(i == 0 and existed_featurecombinations.shape[0] == 1):
                existed_featurecombinations[i,:] = random_column_indices.T
            else:
                existed_featurecombinations = np.append(existed_featurecombinations, np.expand_dims(random_column_indices.T,axis = 0), axis = 0)
        
            # feature selection
            X_train_select = X_train[:, random_column_indices]  
            X_test_select = X_test[:, random_column_indices]    
            accs=[]   
            # train and test models
            for name, model in models.items():
                model.fit(X_train_select, y_train)  # model training
                y_pred = model.predict(X_test_select)  # model prediction
                acc = accuracy_score(y_test, y_pred)
                accs.append(acc)
            save_featurecombiantion_result(csv_result_file, accs, np.sort(random_column_indices.T))
            
def compute_metrics_for_combinations_for_classification(X_metric_use, y_metric_use, X_org,
                                                        num_sets, num_total_features, result_path, metric_path, is_overwrite_metric = True):
                                                    
    """
    Computes a variety of feature selection and evaluation metrics for different combinations of features
    in a classification setting, and saves the results to CSV files.
    This function precomputes several feature metrics (such as variance, entropy, correlation, etc.) for all features,
    and then, for each specified number of selected features, iterates over all possible feature combinations
    (as previously saved in result files). For each combination, it computes and aggregates the relevant metrics,
    and saves the results to a metrics CSV file.
    Parameters
    ----------
    X_metric_use : np.ndarray
        The feature matrix to use for metric computation (samples x features).
    y_metric_use : np.ndarray
        The target labels corresponding to X_metric_use.
    X_org : np.ndarray
        The original feature matrix (may be used for certain metrics).
    num_sets : list or iterable of int
        List of numbers of features to select for each combination (e.g., [5, 10, 20]).
    num_total_features : int
        The total number of features in the dataset.
    result_path : str
        Path prefix for the CSV files containing feature combinations and results.
    metric_path : str
        Path prefix for the CSV files where computed metrics will be saved.
    is_overwrite_metric : bool, optional (default=True)
        Whether to overwrite existing metric files or append to them.
    Notes
    -----
    - The function expects that the feature combinations for each `num_select` are already saved in CSV files.
    - For each combination, a variety of metrics are computed, including variance, entropy, correlation, Laplacian score,
        ReliefF, ANOVA F-value, mutual information, MCFS, MRMR, ERFS, and others.
    - The function is designed to handle large datasets efficiently by subsampling for certain metrics if needed.
    - Results are saved incrementally to CSV files to allow for resuming computation.
    Returns
    -------
    None
        The function saves computed metrics to disk and does not return any value.
    """
 
    covariance_matrix = np.cov(X_metric_use, rowvar = False)#covariance_matrix
    similarity_matrix = np.abs(cosine_similarity(X_metric_use.T))#similarity matrix
    correlation_matrix = X_metric_use.T @ X_metric_use #automatic correlation matrix
    similarity_matrix [similarity_matrix > 1] = 1
    F_values, _ = f_classif(X_metric_use, y_metric_use) #f_classif algorithm's F values pre-computed
    corr_matrix = np.abs(np.corrcoef(X_metric_use, rowvar = False))#correlation matrix
    corr_matrix = np.interp(corr_matrix, (0, 1), (0.3, 1))# stretch the range of correlation coefficients to prevent the influence of very small values
    yks, x_corr, xty = mcfs_parameter(X_metric_use, k_clusters = 2) #pre-compute parameters required for MCBS evaluation
    
    inf_gai_for_all_features = np.array(mutual_info_classif(X_metric_use, y_metric_use, random_state = 42))
    anv_val_for_all_features = np.array(f_classif(X_metric_use, y_metric_use))[0,:]        
    
    rlf_fser = ReliefF(n_neighbors = min(100, np.ceil(0.3*X_metric_use.shape[0])),  discrete_threshold = 20,
                        n_features_to_select = num_total_features, n_jobs = 20)
    if(X_metric_use.shape[0] > 3000):
        subset = np.random.choice(X_metric_use.shape[0], 3000, replace=False)
        X_sub = X_metric_use[subset, :]
        y_sub = y_metric_use[subset]
        rlf_fser.fit(X_sub, y_sub)
    else:
        rlf_fser.fit(X_metric_use, y_metric_use)
    rlf_val_for_all_features = rlf_fser.feature_importances_
    
    metrics_names = ['Variance', 'Sim Entropy', 'Rep Entropy', 'SPECtrum','Lap Score', 
                        'Mutual Corre', 'MICI', 'MDCM', 'Linear Depend','MPMR', 'MCFS', 'RSPCA',
                        'ANOVA F', 'ReliefF','Info Gain', 'MRMR', 'ERFS']
    ######## feature metrics computation for all features##########
    lap_scr_for_all_features = laplacian_score(X_metric_use)
    sim_ent_for_all_features = similarity_entropy(X_metric_use, similarity_matrix = similarity_matrix)
    svd_ent_for_all_features = svd_entropy(X_metric_use, convariance_matrix = covariance_matrix)
    var_scr_for_all_features = variance_score(X_org)
    spc_ent_for_all_features = SPEC.spec(X_metric_use, style = 0)####
    
    for num_select in num_sets:
        print(f'metric computing for {num_select} features...')
        if num_select == num_total_features:
            continue
        ########### Read saved results, check if the current feature combination has been computed ##########
        csv_result_file = f'{result_path}_classification_{num_select}.csv'###result file
        csv_metric_file = f'{metric_path}_classification_{num_select}.csv'###metric file
        if not os.path.exists(csv_result_file):
            print(f'Cannot find {csv_result_file}')
            continue
        featurecombinations = get_featureconbinations_from_csv(csv_result_file, num_select, num_accurices_cols = 5)
        total_rows = featurecombinations.shape[0]
        if os.path.exists(csv_metric_file):
            if is_overwrite_metric:
                clear_CSV_keepheader(csv_metric_file)
                num_rows_computed = 0
            else:
                num_rows_computed = get_num_conputed_metrics_from_csv(csv_metric_file)
        else:
            num_rows_computed = 0
            
        num_rows_to_compute = total_rows - num_rows_computed
        curr_row = num_rows_computed
        
        for i in range(num_rows_to_compute):
            
            selected_indices = featurecombinations[curr_row, :]
            lap_scr = lap_scr_for_all_features[selected_indices].sum()
            sim_ent = sim_ent_for_all_features[selected_indices].sum()
            svd_ent = svd_ent_for_all_features[selected_indices].sum()
            var_src = var_scr_for_all_features[selected_indices].sum()
            spc_ent = spc_ent_for_all_features[selected_indices].sum()
            anv_val = anv_val_for_all_features[selected_indices].sum()
            rlf_val = rlf_val_for_all_features[selected_indices].sum()
            inf_gai = inf_gai_for_all_features[selected_indices].sum()
            
            ############# feature selection metrics computation for selected features ##########
            mut_cor = mutual_corre(X_metric_use,selected_indices,similarity_matrix = similarity_matrix)
            mici = max_info_compress_index(X_metric_use,selected_indices,similarity_matrix = similarity_matrix)
            mdcm_eig,lin_dep = max_determinant_of_covmatrix_and_linear_dependency(X_metric_use, selected_indices,convariance_matrix = covariance_matrix)
            lin_rep = linear_rep_error(X_metric_use, selected_indices)
            rsp_val = rspca_value(X_metric_use, selected_indices)
            mrmr_v = mrmr_value(selected_indices, F_values, corr_matrix)
            mcfs_s = mcfs_score(X_metric_use, yks, x_corr, xty, selected_cols = selected_indices)
            erf_val = erfs_error(X_metric_use, y_metric_use, selected_cols = selected_indices, correlation_matrix = correlation_matrix)
            
            metrics = [var_src, sim_ent, svd_ent, lap_scr, spc_ent, mut_cor, 
                        mici, mdcm_eig,  lin_dep, lin_rep, mcfs_s, rsp_val,
                        anv_val, rlf_val, inf_gai, mrmr_v, erf_val]
            save_metric_result(csv_metric_file, metrics, header = metrics_names)
            curr_row += 1
def fs_methods_evaluation_for_classification(X_fsmethod_use, X_test, X_org, y_fsmethod_use, y_test, 
                                            num_sets, num_total_features, models, result_path):
    """
    Evaluate multiple feature selection methods for classification tasks and compute accuracy metrics for various models.
    This function applies a suite of feature selection algorithms to the provided training data, selects different numbers
    of features as specified, and evaluates the classification accuracy of several machine learning models by NA metric on the selected
    features. The results are saved as CSV files for each feature set size.
    Parameters
    ----------
    X_fsmethod_use : np.ndarray
        Feature matrix for training, after initial feature selection or preprocessing.
    X_test : np.ndarray
        Feature matrix for testing.
    y_fsmethod_use : np.ndarray or pd.Series
        Target labels for training set.
    y_test : np.ndarray or pd.Series
        Target labels for test set.
    num_sets : np.ndarray or list
        Array or list of integers specifying the different numbers of features to select and evaluate.
    num_total_features : int
        Total number of features in the dataset.
    models : dict
        Dictionary of model name to instantiated scikit-learn classifier objects.
    result_path : str
        Base path for saving the CSV result files.
    Returns
    -------
    None
        The function saves accuracy results to CSV files for each feature set size and does not return any value.
    Notes
    -----
    - The function evaluates a wide range of feature selection methods, including variance, entropy-based, Laplacian score,
        SPECtrum, mutual correlation, MICI, MDCM, linear dependency, MPMR, MCFS, RSPCA, ANOVA F, ReliefF, Info Gain, MRMR, and ERFS.
    - For each method and feature set size, the function evaluates classification accuracy using multiple models (SVM, Random Forest,
        Logistic Regression, KNN, Decision Tree).
    - Results are saved as CSV files with accuracy scores for each method-model combination.
    """
    covariance_matrix = np.cov(X_fsmethod_use, rowvar = False)# covariance matrix
    similarity_matrix = np.abs(cosine_similarity(X_fsmethod_use.T))# similarity matrix
    correlation_matrix = X_fsmethod_use.T @ X_fsmethod_use # automatic correlation matrix
    similarity_matrix [similarity_matrix > 1] = 1
    F_values, _ = f_classif(X_fsmethod_use, y_fsmethod_use) # f_classif algorithm's F values pre-computed
    corr_matrix = np.abs(np.corrcoef(X_fsmethod_use, rowvar = False))# correlation matrix
    corr_matrix = np.interp(corr_matrix, (0, 1), (0.3, 1))# stretch the range of correlation coefficients to prevent the influence of very small values
    
    inf_gai_for_all_features = np.array(mutual_info_classif(X_fsmethod_use, y_fsmethod_use, random_state = 42))
    anv_val_for_all_features = np.array(f_classif(X_fsmethod_use, y_fsmethod_use))[0,:]
    
    rlf_fser = ReliefF(n_neighbors = min(100, np.ceil(0.3*X_fsmethod_use.shape[0])), discrete_threshold = 20,
                        n_features_to_select = num_total_features, n_jobs = 20)
    if(X_fsmethod_use.shape[0] > 3000):
        subset = np.random.choice(X_fsmethod_use.shape[0], 3000, replace=False)
        X_sub = X_fsmethod_use[subset, :]
        y_sub = y_fsmethod_use[subset]
        rlf_fser.fit(X_sub, y_sub)
    else:
        rlf_fser.fit(X_fsmethod_use, y_fsmethod_use)
    rlf_val_for_all_features = rlf_fser.feature_importances_
    
    lap_scr_for_all_features = laplacian_score(X_fsmethod_use)
    sim_ent_for_all_features = similarity_entropy(X_fsmethod_use, similarity_matrix = similarity_matrix)
    svd_ent_for_all_features = svd_entropy(X_fsmethod_use, convariance_matrix = covariance_matrix)
    var_scr_for_all_features = variance_score(X_org)
    spc_ent_for_all_features = SPEC.spec(X_fsmethod_use, style = 0)
        
    max_selected_num = np.max(num_sets) - 1  
    mrm_selected_result, relv, coff = mrmr_classif(X = pd.DataFrame(X_fsmethod_use), y = pd.DataFrame(y_fsmethod_use), 
                                                   K = max_selected_num, return_scores = True)
    mcf_selected_result = multi_cluster_fs(X_fsmethod_use)
    lap_selected_result = np.argsort(lap_scr_for_all_features)
    var_selected_result = np.argsort(var_scr_for_all_features)[::-1]
    sim_selected_result = np.argsort(sim_ent_for_all_features)[::-1]
    svd_selected_result = np.argsort(svd_ent_for_all_features) 
    spc_selected_result = np.argsort(spc_ent_for_all_features)
    rlf_selected_result = np.argsort(rlf_val_for_all_features)[::-1]
    ifg_selected_result = np.argsort(inf_gai_for_all_features)[::-1]
    anf_selected_result = np.argsort(anv_val_for_all_features)[::-1]

    
    for num_selected_feature in num_sets:
        print(f'choosing {num_selected_feature} features...')
        if(num_selected_feature == num_total_features):
            continue
        csv_fsmethod_file = f'{result_path}_fsmethod_{num_selected_feature}.csv'
        cur_lap_selected_result = lap_selected_result[:num_selected_feature]
        cur_var_selected_result = var_selected_result[:num_selected_feature]
        cur_sim_selected_result = sim_selected_result[:num_selected_feature]
        cur_svd_selected_result = svd_selected_result[:num_selected_feature]
        cur_spc_selected_result = spc_selected_result[:num_selected_feature]
        cur_anf_selected_result = anf_selected_result[:num_selected_feature]
        cur_mrm_selected_result = mrm_selected_result[:num_selected_feature]
        cur_rlf_selected_result = rlf_selected_result[:num_selected_feature]
        cur_ifg_selected_result = ifg_selected_result[:num_selected_feature]
        cur_mcf_selected_result = mcf_selected_result[:num_selected_feature]
        
        cur_mic_selected_result = mici_fs(X_fsmethod_use, num_selected_feature = num_selected_feature, similarity_matrix = similarity_matrix)
        cur_mtc_selected_result = mutual_corre_fs(X_fsmethod_use, num_selected_feature = num_selected_feature,similarity_matrix = similarity_matrix)
        cur_mdc_selected_result = mdcm_fs(X_fsmethod_use, num_selected_feature = num_selected_feature,covariance_matrix = covariance_matrix) 
        cur_ldp_selected_result = linear_dependency_fs(X_fsmethod_use,num_selected_feature=num_selected_feature, covariance_matrix = covariance_matrix)
        cur_mpm_selected_result = mpmr_fs(X_fsmethod_use,num_selected_feature=num_selected_feature,correlatin_matrix = correlation_matrix)
        _, cur_rsp_selected_result,_ = rs_pca(X_fsmethod_use.T, sigma = 0.1, num_selected_feature = num_selected_feature)
        cur_mre_selected_result, _ = min_representation_err_fs(X_fsmethod_use, y_fsmethod_use, num_selected_feature)

        fs_method_selected_results = {
            'Variance'        : cur_var_selected_result,
            'Sim Entropy'     : cur_sim_selected_result,
            'Rep Entropy'     : cur_svd_selected_result,
            'Lap Score'       : cur_lap_selected_result,
            'SPECtrum'        : cur_spc_selected_result,
            'Mutual Corre'    : cur_mtc_selected_result,
            'MICI'            : cur_mic_selected_result,
            'MDCM'            : cur_mdc_selected_result,
            'Linear Depend'   : cur_ldp_selected_result,
            'MPMR'            : cur_mpm_selected_result,
            'MCFS'            : cur_mcf_selected_result,    
            'RSPCA'           : cur_rsp_selected_result,         
            'ANOVA F'         : cur_anf_selected_result,
            'ReliefF'         : cur_rlf_selected_result,
            'Info Gain'       : cur_ifg_selected_result,
            'MRMR'            : cur_mrm_selected_result,        
            'ERFS'            : cur_mre_selected_result,
        }
        fs_method_names=[]
        row = 0
        fs_acc = np.zeros((len(fs_method_selected_results), len(models)), dtype = float)
        for fsmethod, selectedresult in fs_method_selected_results.items():
            X_train_select = X_fsmethod_use[:, selectedresult]
            X_test_select = X_test[:, selectedresult]
            col = 0
            fs_method_names.append(fsmethod)
            for name, model in models.items():
                model.fit(X_train_select, y_fsmethod_use)  # model training
                y_pred = model.predict(X_test_select)  # 
                acc = accuracy_score(y_test, y_pred)
                fs_acc[row, col] = acc
                col += 1
            row += 1
        df = pd.DataFrame(fs_acc, index = fs_method_names, columns = ['SVM','randomfroest','logesticregression','knn','decisiontree'] )
        df.to_csv(csv_fsmethod_file)
        
def random_feature_selection_regression(X_train, X_test, y_train, y_test, 
                                        num_sets, num_total_features, models, 
                                        result_path, new_iter_times = 50, max_iter_times = 500):
    """
    Performs random feature selection for regression tasks and evaluates model accuracy.
    For each specified number of features to select, this function randomly samples unique feature subsets,
    trains and evaluates the provided regression models on these subsets, and saves the results.
    It avoids recomputation by checking previously saved results.
    Args:
        X_train (np.ndarray): Training feature matrix of shape (n_samples, n_features).
        X_test (np.ndarray): Test feature matrix of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        num_sets (list or iterable): List of integers specifying the number of features to select in each experiment.
        num_total_features (int): Total number of available features.
        models (dict): Dictionary mapping model names to sklearn-like classifier instances.
        result_path (str): Base path for saving result CSV files.
        new_iter_times (int, optional): Number of new random feature combinations to try per setting. Default is 50.
        max_iter_times (int, optional): Maximum number of total iterations (feature combinations) per setting. Default is 500.
    Notes:
        - If the number of features to select equals the total number of features, only one iteration is performed.
        - If the number of possible combinations is less than `max_iter_times`, all combinations are tried.
        - Previously computed feature combinations are loaded from CSV files to avoid duplication.
        - Results are saved incrementally to CSV files named according to the number of features selected.
    Returns:
        None. Results are saved to disk.
    """        
    for num_select in num_sets:
        _iter_times = new_iter_times# real loop times
        max_iter_times_curr = max_iter_times
        num_columns_to_select = num_select
        print(f'\nprocessing choosing {num_select} features...', end =' ', flush = True)
        if num_columns_to_select == num_total_features:
            max_iter_times = 1
            _iter_times = 1
        if num_columns_to_select < 5 or num_total_features - num_columns_to_select < 5 :
            max_iter_times_curr = min(math.comb(num_total_features, num_select), max_iter_times)
        ########### Read saved results, check if the current feature combination has been computed ##########

        csv_result_file = f'{result_path}_regression_{num_select}.csv'
        if os.path.exists(csv_result_file):
            existed_featurecombinations = get_featureconbinations_from_csv(csv_result_file, num_columns_to_select, num_accurices_cols = 5)
            if(existed_featurecombinations.shape[0] > max_iter_times):
                continue
            _iter_times = min(new_iter_times, max_iter_times_curr - existed_featurecombinations.shape[0])
        else:
            _iter_times = min(new_iter_times, max_iter_times_curr)
            existed_featurecombinations = np.zeros((1,num_columns_to_select),dtype=int)
            np.expand_dims(existed_featurecombinations, axis = 0)

        for i in range(_iter_times):
            print(i + 1, end =' ',flush = True)
            random_column_indices = np.sort(np.random.choice(num_total_features, size = num_columns_to_select, replace = False))
            if(existed_featurecombinations.shape[0] > 1):# check if the random feature combination already exists
                while check_featurecombination_exist(existed_featurecombinations, random_column_indices):
                    random_column_indices = np.sort(np.random.choice(num_total_features, size = num_columns_to_select, replace = False))
            if(i == 0 and existed_featurecombinations.shape[0] == 1):
                existed_featurecombinations[i,:] = random_column_indices.T
            else:
                existed_featurecombinations = np.append(existed_featurecombinations, np.expand_dims(random_column_indices.T,axis = 0), axis = 0)
        
            # feature selection
            X_train_select = X_train[:, random_column_indices]  
            X_test_select = X_test[:, random_column_indices]    
            accs=[]   
            # main loop for model training and prediction
            for name, model in models.items():
                model.fit(X_train_select, y_train)  # model training
                y_pred = model.predict(X_test_select)  # model prediction
                acc = 1 - np.linalg.norm(y_test - y_pred) / (np.linalg.norm(y_test) + 1e-10)#Regression accuracy metric RA
                accs.append(acc)
            save_featurecombiantion_result(csv_result_file, accs, np.sort(random_column_indices.T))
            
def compute_metrics_for_combinations_for_regression(X_metric_use, X_org, y_metric_use, 
                                     num_sets, num_total_features, 
                                     result_path, metric_path, is_overwrite_metric = True):
    """
    Computes a variety of feature selection and evaluation metrics for different combinations of features
    in a classification setting, and saves the results to CSV files.
    This function precomputes several feature metrics (such as variance, entropy, correlation, etc.) for all features,
    and then, for each specified number of selected features, iterates over all possible feature combinations
    (as previously saved in result files). For each combination, it computes and aggregates the relevant metrics,
    and saves the results to a metrics CSV file.
    Parameters
    ----------
    X_metric_use : np.ndarray
        The feature matrix to use for metric computation (samples x features).
    y_metric_use : np.ndarray
        The target labels corresponding to X_metric_use.
    X_org : np.ndarray
        The original feature matrix (may be used for certain metrics).
    num_sets : list or iterable of int
        List of numbers of features to select for each combination (e.g., [5, 10, 20]).
    num_total_features : int
        The total number of features in the dataset.
    result_path : str
        Path prefix for the CSV files containing feature combinations and results.
    metric_path : str
        Path prefix for the CSV files where computed metrics will be saved.
    is_overwrite_metric : bool, optional (default=True)
        Whether to overwrite existing metric files or append to them.
    Notes
    -----
    - The function expects that the feature combinations for each `num_select` are already saved in CSV files.
    - For each combination, a variety of metrics are computed, including variance, entropy, correlation, Laplacian score,
        ReliefF, ANOVA F-value, mutual information, MCFS, MRMR, ERFS, and others.
    - The function is designed to handle large datasets efficiently by subsampling for certain metrics if needed.
    - Results are saved incrementally to CSV files to allow for resuming computation.
    Returns
    -------
    None
        The function saves computed metrics to disk and does not return any value.
    """
    covariance_matrix = np.cov(X_metric_use, rowvar = False)#covariance_matrix
    similarity_matrix = np.abs(cosine_similarity(X_metric_use.T))#similarity matrix
    correlation_matrix = X_metric_use.T @ X_metric_use #automatic correlation matrix
    similarity_matrix [similarity_matrix > 1] = 1
    F_values, _ = f_regression(X_metric_use, y_metric_use) #f_regression algorithm's F values pre-computed
    corr_matrix = np.abs(np.corrcoef(X_metric_use, rowvar = False))#correlation matrix
    corr_matrix = np.interp(corr_matrix, (0, 1), (0.3, 1))# stretch the range of correlation coefficients to prevent the influence of very small values
    yks, x_corr, xty = mcfs_parameter(X_metric_use, k_clusters = 2) #pre-compute parameters required for MCBS evaluation
    
    
    inf_gai_for_all_features = np.array(mutual_info_regression(X_metric_use, y_metric_use, random_state = 42))
    anv_val_for_all_features = np.array(f_regression(X_metric_use, y_metric_use))[0,:]        
    
    rlf_fser = ReliefF(n_neighbors = min(100, np.ceil(0.3*X_metric_use.shape[0])),  discrete_threshold = 20,
                        n_features_to_select = num_total_features, n_jobs = 20)
    if(X_metric_use.shape[0] > 3000):
        subset = np.random.choice(X_metric_use.shape[0], 3000, replace=False)
        X_sub = X_metric_use[subset, :]
        y_sub = y_metric_use[subset]
        rlf_fser.fit(X_sub, y_sub)
    else:
        rlf_fser.fit(X_metric_use, y_metric_use)
    rlf_val_for_all_features = rlf_fser.feature_importances_
    
    metrics_names = ['Variance', 'Sim Entropy', 'Rep Entropy', 'SPECtrum','Lap Score', 
                        'Mutual Corre', 'MICI', 'MDCM', 'Linear Depend','MPMR', 'MCFS', 'RSPCA',
                        'ANOVA F', 'ReliefF','Info Gain', 'MRMR', 'ERFS']
    ######## feature metrics computation for all features##########
    lap_scr_for_all_features = laplacian_score(X_metric_use)
    sim_ent_for_all_features = similarity_entropy(X_metric_use, similarity_matrix = similarity_matrix)
    svd_ent_for_all_features = svd_entropy(X_metric_use, convariance_matrix = covariance_matrix)
    var_scr_for_all_features = variance_score(X_org)
    spc_ent_for_all_features = SPEC.spec(X_metric_use, style = 0)####样本量很大时，计算会崩溃!!!!
    
    for num_select in num_sets:
        print(f'metric computing for {num_select} features...')
        if num_select == num_total_features:
            continue
        ########### Read saved results, check if the current feature combination has been computed ##########
        csv_result_file = f'{result_path}_regression_{num_select}.csv'
        csv_metric_file = f'{metric_path}_regression_{num_select}.csv'
        if not os.path.exists(csv_result_file):
            print(f'Could not find{csv_result_file}')
            continue
        featurecombinations = get_featureconbinations_from_csv(csv_result_file, num_select, num_accurices_cols = 5)
        total_rows = featurecombinations.shape[0]
        if os.path.exists(csv_metric_file):
            if is_overwrite_metric:
                clear_CSV_keepheader(csv_metric_file)
                num_rows_computed = 0
            else:
                num_rows_computed = get_num_conputed_metrics_from_csv(csv_metric_file)
        else:
            num_rows_computed = 0
            
        num_rows_to_compute = total_rows - num_rows_computed
        curr_row = num_rows_computed
        
        for i in range(num_rows_to_compute):
            
            selected_indices = featurecombinations[curr_row, :]
            lap_scr = lap_scr_for_all_features[selected_indices].sum()
            sim_ent = sim_ent_for_all_features[selected_indices].sum()
            svd_ent = svd_ent_for_all_features[selected_indices].sum()
            var_src = var_scr_for_all_features[selected_indices].sum()
            spc_ent = spc_ent_for_all_features[selected_indices].sum()
            anv_val = anv_val_for_all_features[selected_indices].sum()
            rlf_val = rlf_val_for_all_features[selected_indices].sum()
            inf_gai = inf_gai_for_all_features[selected_indices].sum()
            
            
            ###### 以下指标需要根据特征组合的不同，一组特征组合调用一次################
            mut_cor = mutual_corre(X_metric_use,selected_indices,similarity_matrix = similarity_matrix)
            mici = max_info_compress_index(X_metric_use,selected_indices,similarity_matrix = similarity_matrix)
            mdcm_eig,lin_dep = max_determinant_of_covmatrix_and_linear_dependency(X_metric_use, selected_indices,convariance_matrix = covariance_matrix)
            lin_rep = linear_rep_error(X_metric_use, selected_indices)
            rsp_val = rspca_value(X_metric_use, selected_indices)
            mrmr_v = mrmr_value(selected_indices, F_values, corr_matrix)
            mcfs_s = mcfs_score(X_metric_use, yks, x_corr, xty, selected_cols = selected_indices)
            erf_val = erfs_error(X_metric_use, y_metric_use, selected_cols = selected_indices, correlation_matrix = correlation_matrix)
            
            metrics = [var_src, sim_ent, svd_ent, lap_scr, spc_ent, mut_cor, 
                        mici, mdcm_eig,  lin_dep, lin_rep, mcfs_s, rsp_val,
                        anv_val, rlf_val, inf_gai, mrmr_v, erf_val]
            save_metric_result(csv_metric_file, metrics, header = metrics_names)
            curr_row += 1
def fs_methods_evaluation_for_regression(X_fsmethod_use, X_test,X_org, y_fsmethod_use, y_test, 
                                            num_sets, num_total_features, models, result_path):
    """
    Evaluate multiple feature selection methods for regression tasks and compute accuracy metrics for various models.
    This function applies a suite of feature selection algorithms to the provided training data, selects different numbers
    of features as specified, and evaluates the regression accuracy of several machine learning models by NA metric on the selected
    features. The results are saved as CSV files for each feature set size.
    Parameters
    ----------
    X_fsmethod_use : np.ndarray
        Feature matrix for training, after initial feature selection or preprocessing.
    X_test : np.ndarray
        Feature matrix for testing.
    y_fsmethod_use : np.ndarray or pd.Series
        Target labels for training set.
    y_test : np.ndarray or pd.Series
        Target labels for test set.
    num_sets : np.ndarray or list
        Array or list of integers specifying the different numbers of features to select and evaluate.
    num_total_features : int
        Total number of features in the dataset.
    models : dict
        Dictionary of model name to instantiated scikit-learn classifier objects.
    result_path : str
        Base path for saving the CSV result files.
    Returns
    -------
    None
        The function saves accuracy results to CSV files for each feature set size and does not return any value.
    Notes
    -----
    - The function evaluates a wide range of feature selection methods, including variance, entropy-based, Laplacian score,
        SPECtrum, mutual correlation, MICI, MDCM, linear dependency, MPMR, MCFS, RSPCA, ANOVA F, ReliefF, Info Gain, MRMR, and ERFS.
    - For each method and feature set size, the function evaluates classification accuracy using multiple models (SVM, Random Forest,
        Logistic Regression, KNN, Decision Tree).
    - Results are saved as CSV files with accuracy scores for each method-model combination.
    """
    covariance_matrix = np.cov(X_fsmethod_use, rowvar = False)#convariance matrix
    similarity_matrix = np.abs(cosine_similarity(X_fsmethod_use.T))#similarity matrix
    correlation_matrix = X_fsmethod_use.T @ X_fsmethod_use #autocorrelation matrix
    similarity_matrix [similarity_matrix > 1] = 1
    F_values, _ = f_regression(X_fsmethod_use, y_fsmethod_use) #MRMR algorithm's F values pre-computed
    corr_matrix = np.abs(np.corrcoef(X_fsmethod_use, rowvar = False))#correlation matrix
    corr_matrix = np.interp(corr_matrix, (0, 1), (0.3, 1))# stretch the range of correlation coefficients to prevent the influence of very small values
    
    inf_gai_for_all_features = np.array(mutual_info_regression(X_fsmethod_use, y_fsmethod_use, random_state = 42))
    anv_val_for_all_features = np.array(f_regression(X_fsmethod_use, y_fsmethod_use))[0,:]
    
    rlf_fser = ReliefF(n_neighbors = min(100, np.ceil(0.3*X_fsmethod_use.shape[0])), discrete_threshold = 20,
                        n_features_to_select = num_total_features, n_jobs = 20)
    if(X_fsmethod_use.shape[0] > 3000):
        subset = np.random.choice(X_fsmethod_use.shape[0], 3000, replace=False)
        X_sub = X_fsmethod_use[subset, :]
        y_sub = y_fsmethod_use[subset]
        rlf_fser.fit(X_sub, y_sub)
    else:
        rlf_fser.fit(X_fsmethod_use, y_fsmethod_use)
    rlf_val_for_all_features = rlf_fser.feature_importances_
    
    lap_scr_for_all_features = laplacian_score(X_fsmethod_use)
    sim_ent_for_all_features = similarity_entropy(X_fsmethod_use, similarity_matrix = similarity_matrix)
    svd_ent_for_all_features = svd_entropy(X_fsmethod_use, convariance_matrix = covariance_matrix)
    var_scr_for_all_features = variance_score(X_org)
    spc_ent_for_all_features = SPEC.spec(X_fsmethod_use, style = 0)
        
    max_selected_num = np.max(num_sets) - 1 
    mrm_selected_result, relv, coff = mrmr_regression(X = pd.DataFrame(X_fsmethod_use), y = pd.DataFrame(y_fsmethod_use), K = max_selected_num, return_scores = True)
    mcf_selected_result = multi_cluster_fs(X_fsmethod_use)
    lap_selected_result = np.argsort(lap_scr_for_all_features)
    var_selected_result = np.argsort(var_scr_for_all_features)[::-1]
    sim_selected_result = np.argsort(sim_ent_for_all_features)[::-1]
    svd_selected_result = np.argsort(svd_ent_for_all_features) 
    spc_selected_result = np.argsort(spc_ent_for_all_features)
    rlf_selected_result = np.argsort(rlf_val_for_all_features)[::-1]
    ifg_selected_result = np.argsort(inf_gai_for_all_features)[::-1]
    anf_selected_result = np.argsort(anv_val_for_all_features)[::-1]

    
    for num_selected_feature in num_sets:
        print(f'choosing {num_selected_feature} features...')
        if(num_selected_feature == num_total_features):
            continue
        csv_fsmethod_file = f'{result_path}_fsmethod_{num_selected_feature}.csv'
        cur_lap_selected_result = lap_selected_result[:num_selected_feature]
        cur_var_selected_result = var_selected_result[:num_selected_feature]
        cur_sim_selected_result = sim_selected_result[:num_selected_feature]
        cur_svd_selected_result = svd_selected_result[:num_selected_feature]
        cur_spc_selected_result = spc_selected_result[:num_selected_feature]
        cur_anf_selected_result = anf_selected_result[:num_selected_feature]
        cur_mrm_selected_result = mrm_selected_result[:num_selected_feature]
        cur_rlf_selected_result = rlf_selected_result[:num_selected_feature]
        cur_ifg_selected_result = ifg_selected_result[:num_selected_feature]
        cur_mcf_selected_result = mcf_selected_result[:num_selected_feature]
        
        cur_mic_selected_result = mici_fs(X_fsmethod_use, num_selected_feature = num_selected_feature, similarity_matrix = similarity_matrix)
        cur_mtc_selected_result = mutual_corre_fs(X_fsmethod_use, num_selected_feature = num_selected_feature,similarity_matrix = similarity_matrix)
        cur_mdc_selected_result = mdcm_fs(X_fsmethod_use, num_selected_feature = num_selected_feature,covariance_matrix = covariance_matrix) 
        cur_ldp_selected_result = linear_dependency_fs(X_fsmethod_use,num_selected_feature=num_selected_feature, covariance_matrix = covariance_matrix)
        cur_mpm_selected_result = mpmr_fs(X_fsmethod_use,num_selected_feature=num_selected_feature,correlatin_matrix = correlation_matrix)

        
        _, cur_rsp_selected_result,_ = rs_pca(X_fsmethod_use.T, sigma = 0.1, num_selected_feature = num_selected_feature)
        cur_mre_selected_result, _ = min_representation_err_fs(X_fsmethod_use, y_fsmethod_use, num_selected_feature)

        fs_method_selected_results = {
            'Variance'        : cur_var_selected_result,
            'Sim Entropy'     : cur_sim_selected_result,
            'Rep Entropy'     : cur_svd_selected_result,
            'Lap Score'       : cur_lap_selected_result,
            'SPECtrum'        : cur_spc_selected_result,
            'Mutual Corre'    : cur_mtc_selected_result,
            'MICI'            : cur_mic_selected_result,
            'MDCM'            : cur_mdc_selected_result,
            'Linear Depend'   : cur_ldp_selected_result,
            'MPMR'            : cur_mpm_selected_result,
            'MCFS'            : cur_mcf_selected_result,    
            'RSPCA'           : cur_rsp_selected_result,         
            'ANOVA F'         : cur_anf_selected_result,
            'ReliefF'         : cur_rlf_selected_result,
            'Info Gain'       : cur_ifg_selected_result,
            'MRMR'            : cur_mrm_selected_result,        
            'ERFS'            : cur_mre_selected_result,
        }
        fs_method_names=[]
        row = 0
        fs_acc = np.zeros((len(fs_method_selected_results), len(models)), dtype = float)
        for fsmethod, selectedresult in fs_method_selected_results.items():
            X_train_select = X_fsmethod_use[:, selectedresult]
            X_test_select = X_test[:, selectedresult]
            col = 0
            fs_method_names.append(fsmethod)
            for name, model in models.items():
                model.fit(X_train_select, y_fsmethod_use)  # model training
                y_pred = model.predict(X_test_select)  # model prediction
                acc = 1 - np.linalg.norm(y_test - y_pred) / (np.linalg.norm(y_test) + 1e-10)#Regression accuracy metric RA
                fs_acc[row, col] = acc
                col += 1
            row += 1
        df = pd.DataFrame(fs_acc, index = fs_method_names, columns = ['Linear','ridge','LASSO','Gradient','randomfroest'] )
        df.to_csv(csv_fsmethod_file)
        
def random_feature_selection_clustering(X, y_true, num_sets, num_total_features, models, result_path,
                                        is_multiple_labels = False, multi_lbl_true = None, new_iter_times = 50, max_iter_times = 500): 
                                        
    """
    Performs random feature selection and clustering evaluation for different numbers of selected features.
    For each specified number of features to select, the function randomly samples unique feature subsets,
    applies clustering models, evaluates clustering performance, and saves the results. If results for a given
    feature combination already exist, it avoids recomputation.
    Args:
        X (np.ndarray): The feature matrix of shape (n_samples, n_features).
        y_true (np.ndarray): The ground truth labels for clustering evaluation.
        num_sets (list of int): List containing the number of features to select for each experiment.
        num_total_features (int): Total number of available features in X.
        models (dict): Dictionary of clustering models with model names as keys and model instances as values.
        result_path (str): Path prefix for saving result CSV files.
        is_multiple_labels (bool, optional): Indicates if multiple label sets are used for evaluation. Default is False.
        multi_lbl_true (np.ndarray, optional): Ground truth labels for multi-label evaluation. Required if is_multiple_labels is True.
        new_iter_times (int, optional): Number of new random feature subsets to evaluate per setting. Default is 50.
        max_iter_times (int, optional): Maximum total number of feature combinations to evaluate per setting. Default is 500.
    Notes:
        - If the number of selected features is very low or high, the number of possible combinations is capped.
        - Results are appended to CSV files, and previously evaluated combinations are skipped.
        - For clustering models that may assign noise labels (-1), noise points are reassigned using nearest neighbors.
        - The function relies on several helper functions: get_featureconbinations_from_csv, check_featurecombination_exist,
            evaluate_clustering, cluster_acc, and save_featurecombiantion_result.
    Returns:
        None
    """
    
    for num_select in num_sets:            
        _iter_times = new_iter_times#real loop times
        max_iter_times_curr = max_iter_times
        num_columns_to_select = num_select
        print(f'\nprocessing choosing {num_select} features...', end =' ', flush = True)
        if num_columns_to_select == num_total_features:
            max_iter_times = 1
            _iter_times = 1
        if num_columns_to_select < 5 or num_total_features - num_columns_to_select < 5 :
            max_iter_times_curr = min(math.comb(num_total_features, num_select), max_iter_times)
        ########### read saved results, check if the current feature combination has been computed ##########
        csv_result_file = f'{result_path}_clustering_{num_select}.csv'
        if os.path.exists(csv_result_file):
            existed_featurecombinations = get_featureconbinations_from_csv(csv_result_file, num_columns_to_select, num_accurices_cols = 5)
            if(existed_featurecombinations.shape[0] > max_iter_times):
                continue
            _iter_times = min(new_iter_times, max_iter_times_curr - existed_featurecombinations.shape[0])
        else:
            _iter_times = min(new_iter_times, max_iter_times_curr)
            existed_featurecombinations = np.zeros((1,num_columns_to_select),dtype=int)
            np.expand_dims(existed_featurecombinations, axis = 0)

        for i in range(_iter_times):
            print(i + 1, end =' ',flush = True)
            random_column_indices = np.sort(np.random.choice(num_total_features, size = num_columns_to_select, replace = False))
            if(existed_featurecombinations.shape[0] > 1):#check 
                while check_featurecombination_exist(existed_featurecombinations, random_column_indices):
                    random_column_indices = np.sort(np.random.choice(num_total_features, size = num_columns_to_select, replace = False))
            if(i == 0 and existed_featurecombinations.shape[0] == 1):
                existed_featurecombinations[i,:] = random_column_indices.T
            else:
                existed_featurecombinations = np.append(existed_featurecombinations, np.expand_dims(random_column_indices.T,axis = 0), axis = 0)
        
            # feature selection
            X_select = X[:, random_column_indices]
            accs=[]   
            # training and prediction
            for name, model in models.items():
                clst_labels = model.fit_predict(X_select)
                if is_multiple_labels:
                    if (clst_labels.min() < 0): #
                        if(clst_labels.max() < 0):#
                            clst_labels.fill(0)
                        else:
                            noise_mask = (clst_labels == -1)
                            knn = NearestNeighbors(n_neighbors = 1).fit(X_select[~noise_mask])  
                            _, indices = knn.kneighbors(X_select[noise_mask])  
                            clst_labels[noise_mask] = clst_labels[~noise_mask][indices.flatten()]
                        acc,_,_ = evaluate_clustering(multi_lbl_true, clst_labels)
                else:
                    acc = cluster_acc(y_true, clst_labels)
                accs.append(acc)
            save_featurecombiantion_result(csv_result_file, accs, np.sort(random_column_indices.T))
            
def compute_metrics_for_combinations_for_clustering(X, X_org, num_sets, num_total_features, num_clusters, 
                                                    result_path, metric_path, is_overwrite_metric = True):
    
    """
    Computes various feature selection metrics for all combinations of selected features in a clustering context.
    This function processes multiple feature subset sizes, reading precomputed feature combinations from result files.
    For each combination, it computes a set of feature selection metrics (such as variance, entropy-based scores, 
    spectral scores, and others) and saves the results to a corresponding metrics CSV file. It supports resuming 
    computation and overwriting existing metric files.
    Parameters:
        X (np.ndarray): Feature matrix (samples x features) for which metrics are to be computed.
        X_org (np.ndarray): Original feature matrix, used for certain metrics (e.g., variance).
        num_sets (Iterable[int]): List or iterable of feature subset sizes to process.
        num_total_features (int): Total number of features in the dataset.
        num_clusters (int): Number of clusters, used for clustering-based metrics.
        result_path (str): Path prefix for result CSV files containing feature combinations.
        metric_path (str): Path prefix for output CSV files to store computed metrics.
        is_overwrite_metric (bool, optional): Whether to overwrite existing metric files. Default is True.
    Notes:
        - The function assumes that helper functions such as `mcfs_parameter`, `laplacian_score`, 
            `similarity_entropy`, `svd_entropy`, `variance_score`, `SPEC.spec`, `mutual_corre`, 
            `max_info_compress_index`, `max_determinant_of_covmatrix_and_linear_dependency`, 
            `linear_rep_error`, `rspca_value`, `mcfs_score`, `get_featureconbinations_from_csv`, 
            `clear_CSV_keepheader`, `get_num_conputed_metrics_from_csv`, and `save_metric_result` are defined elsewhere.
        - Metric computation is skipped for subsets equal in size to the total number of features.
        - The function prints status updates and skips missing result files.
    """
 
    covariance_matrix = np.cov(X, rowvar = False)#covariance_matrix
    similarity_matrix = np.abs(cosine_similarity(X.T))#similarity matrix
    similarity_matrix [similarity_matrix > 1] = 1
    yks, x_corr, xty = mcfs_parameter(X, k_clusters = num_clusters) #pre-compute parameters required for MCBS evaluation
    
    metrics_names = ['Variance', 'Sim Entropy', 'Rep Entropy', 'SPECtrum','Lap Score', 
                        'Mutual Corre', 'MICI', 'MDCM', 'Linear Depend','MPMR', 'MCFS', 'RSPCA']
    ######compute feature metrics for all feature subsets##########
    lap_scr_for_all_features = laplacian_score(X)
    sim_ent_for_all_features = similarity_entropy(X, similarity_matrix = similarity_matrix)
    svd_ent_for_all_features = svd_entropy(X, convariance_matrix = covariance_matrix)
    var_scr_for_all_features = variance_score(X_org)
    spc_ent_for_all_features = SPEC.spec(X, style = 0)####
    
    for num_select in num_sets:
        print(f'metric computing for {num_select} features...')
        if num_select == num_total_features:
            continue
        ########### read saved results, check if the current feature combination has been computed ##########
        csv_result_file = f'{result_path}_clustering_{num_select}.csv'
        csv_metric_file = f'{metric_path}_clustering_{num_select}.csv'
        if not os.path.exists(csv_result_file):
            print(f'Could not find {csv_result_file}')
            continue
        featurecombinations = get_featureconbinations_from_csv(csv_result_file, num_select, num_accurices_cols = 5)
        total_rows = featurecombinations.shape[0]
        if os.path.exists(csv_metric_file):
            if is_overwrite_metric:
                clear_CSV_keepheader(csv_metric_file)
                num_rows_computed = 0
            else:
                num_rows_computed = get_num_conputed_metrics_from_csv(csv_metric_file)
        else:
            num_rows_computed = 0
            
        num_rows_to_compute = total_rows - num_rows_computed
        curr_row = num_rows_computed
        
        for i in range(num_rows_to_compute):
            
            selected_indices = featurecombinations[curr_row, :]
            lap_scr = lap_scr_for_all_features[selected_indices].sum()
            sim_ent = sim_ent_for_all_features[selected_indices].sum()
            svd_ent = svd_ent_for_all_features[selected_indices].sum()
            var_src = var_scr_for_all_features[selected_indices].sum()
            spc_ent = spc_ent_for_all_features[selected_indices].sum()
            
            
            ###### compute the following metrics for each feature combination ################
            mut_cor = mutual_corre(X,selected_indices,similarity_matrix = similarity_matrix)
            mici = max_info_compress_index(X,selected_indices,similarity_matrix = similarity_matrix)
            mdcm_eig,lin_dep = max_determinant_of_covmatrix_and_linear_dependency(X, selected_indices,convariance_matrix = covariance_matrix)
            lin_rep = linear_rep_error(X, selected_indices)
            rsp_val = rspca_value(X, selected_indices)
            mcfs_s = mcfs_score(X, yks, x_corr, xty, selected_cols = selected_indices)
            
            metrics = [var_src, sim_ent, svd_ent, lap_scr, spc_ent, mut_cor, 
                        mici, mdcm_eig,  lin_dep, lin_rep, mcfs_s, rsp_val]
            save_metric_result(csv_metric_file, metrics, header = metrics_names)
            curr_row += 1
    

def fs_methods_evaluation_for_clustering(X, X_org, y_true, num_sets, num_total_features, models, num_clusters, 
                                        result_path,is_multiple_labels = False, multi_lbl_true = None):
    
    """
    Evaluates various feature selection (FS) methods for clustering tasks and saves the results.
    This function applies multiple unsupervised feature selection algorithms to the input data,
    selects different numbers of features as specified in `num_sets`, and evaluates the clustering
    performance using several clustering models. The results are saved as CSV files for each feature
    subset size.
    Args:
        X (np.ndarray): The feature matrix after preprocessing or transformation (samples x features).
        X_org (np.ndarray): The original feature matrix (samples x features).
        y_true (np.ndarray): The ground truth labels for clustering evaluation.
        num_sets (list or np.ndarray): List of numbers specifying how many features to select for each evaluation.
        num_total_features (int): The total number of features in the dataset.
        models (dict): Dictionary of clustering models to evaluate, with model names as keys and model instances as values.
        num_clusters (int): The number of clusters to use for clustering-based FS methods.
        result_path (str): The base path for saving the CSV result files.
        is_multiple_labels (bool, optional): Whether the dataset has multiple label sets (multi-label clustering). Default is False.
        multi_lbl_true (np.ndarray, optional): The ground truth multi-labels, required if `is_multiple_labels` is True.
    Returns:
        None
    Side Effects:
        - Saves a CSV file for each feature subset size, containing clustering accuracy for each FS method and clustering model.
    Notes:
        - The function assumes that all required FS and clustering methods are properly imported and available.
        - The clustering models in `models` should implement `fit_predict`.
        - The function handles noise labels (-1) for DBSCAN-like algorithms when `is_multiple_labels` is True.
    """
    
    covariance_matrix = np.cov(X, rowvar = False)#covariance_matrix
    similarity_matrix = np.abs(cosine_similarity(X.T))#similarity matrix
    correlation_matrix = X.T @ X #automatic correlation matrix
    similarity_matrix [similarity_matrix > 1] = 1
  
    lap_scr_for_all_features = laplacian_score(X)
    sim_ent_for_all_features = similarity_entropy(X, similarity_matrix = similarity_matrix)
    svd_ent_for_all_features = svd_entropy(X, convariance_matrix = covariance_matrix)
    var_scr_for_all_features = variance_score(X_org)
    spc_ent_for_all_features = SPEC.spec(X, style = 0)
        
    max_selected_num = np.max(num_sets) - 1  
    
    mcf_selected_result = multi_cluster_fs(X, laplacian_reduction_dim = num_clusters)
    lap_selected_result = np.argsort(lap_scr_for_all_features)
    var_selected_result = np.argsort(var_scr_for_all_features)[::-1]
    sim_selected_result = np.argsort(sim_ent_for_all_features)[::-1]
    svd_selected_result = np.argsort(svd_ent_for_all_features)
    spc_selected_result = np.argsort(spc_ent_for_all_features)

    
    for num_selected_feature in num_sets:
        print(f'choosing {num_selected_feature} features...')
        if(num_selected_feature == num_total_features):
            continue
        csv_fsmethod_file = f'{result_path}_fsmethod_{num_selected_feature}.csv'
        cur_lap_selected_result = lap_selected_result[:num_selected_feature]
        cur_var_selected_result = var_selected_result[:num_selected_feature]
        cur_sim_selected_result = sim_selected_result[:num_selected_feature]
        cur_svd_selected_result = svd_selected_result[:num_selected_feature]
        cur_spc_selected_result = spc_selected_result[:num_selected_feature]
        cur_mcf_selected_result = mcf_selected_result[:num_selected_feature]
        
        cur_mic_selected_result = mici_fs(X, num_selected_feature = num_selected_feature, similarity_matrix = similarity_matrix)
        cur_mtc_selected_result = mutual_corre_fs(X, num_selected_feature = num_selected_feature,similarity_matrix = similarity_matrix)
        cur_mdc_selected_result = mdcm_fs(X, num_selected_feature = num_selected_feature,covariance_matrix = covariance_matrix) 
        cur_ldp_selected_result = linear_dependency_fs(X,num_selected_feature=num_selected_feature, covariance_matrix = covariance_matrix)
        cur_mpm_selected_result = mpmr_fs(X,num_selected_feature=num_selected_feature,correlatin_matrix = correlation_matrix)

        
        _, cur_rsp_selected_result,_ = rs_pca(X.T, sigma = 0.1, num_selected_feature = num_selected_feature)

        fs_method_selected_results = {
            'Variance'        : cur_var_selected_result,
            'Sim Entropy'     : cur_sim_selected_result,
            'Rep Entropy'     : cur_svd_selected_result,
            'Lap Score'       : cur_lap_selected_result,
            'SPECtrum'        : cur_spc_selected_result,
            'Mutual Corre'    : cur_mtc_selected_result,
            'MICI'            : cur_mic_selected_result,
            'MDCM'            : cur_mdc_selected_result,
            'Linear Depend'   : cur_ldp_selected_result,
            'MPMR'            : cur_mpm_selected_result,
            'MCFS'            : cur_mcf_selected_result,    
            'RSPCA'           : cur_rsp_selected_result
        }
        fs_method_names=[]
        row = 0
        fs_acc = np.zeros((len(fs_method_selected_results), len(models)), dtype = float)
        for fsmethod, selectedresult in fs_method_selected_results.items():
            X_select = X[:, selectedresult]
            col = 0
            fs_method_names.append(fsmethod)
            for name, model in models.items():
                clst_labels = model.fit_predict(X_select)  # 训练模型
                if is_multiple_labels:
                    if (clst_labels.min() < 0): #
                        if(clst_labels.max() < 0):#
                            clst_labels.fill(0)
                        else:
                            noise_mask = (clst_labels == -1)
                            knn = NearestNeighbors(n_neighbors = 1).fit(X_select[~noise_mask])  
                            _, indices = knn.kneighbors(X_select[noise_mask])  
                            clst_labels[noise_mask] = clst_labels[~noise_mask][indices.flatten()]
                        acc,_,_ = evaluate_clustering(multi_lbl_true, clst_labels)
                else:
                    acc = cluster_acc(y_true, clst_labels)
                fs_acc[row, col] = acc
                col += 1
            row += 1
        df = pd.DataFrame(fs_acc, index = fs_method_names, columns = ['KNN','AGG','SPEC','DBscan','BRICH'] )
        df.to_csv(csv_fsmethod_file)