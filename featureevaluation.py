# featureevaluation.py
"""
featureevaluation.py
This module provides tools for evaluating feature selection methods and feature evaluation metrics
across different datasets, supporting classification, clustering, and regression tasks.
It primarily processes CSV files containing accuracy scores and metric values, computes 
correlation statistics (Spearman and Kendall) to evaluate the relationship between metric scores 
and model performance, and normalizes accuracy scores of feature selection methods.
Main Functions:
---------------
- load_clustering_result(file_path: str): 
    Load clustering results from a CSV file.
- load_classification_result(file_path: str):
    Load classification results from a CSV file.
- load_fsmethod_result(file_path: str):
    Load feature selection method results from a CSV file.
- load_metric_result(file_path: str):
    Load evaluation metric results from a CSV file.
- compute_CI(accs: np.ndarray, metrics: np.ndarray):
    Compute Kendall and Spearman correlation between model accuracies and metric values.
- clear_CSV_keepheader(file_path: str):
    Clear a CSV file but keep its header.
- save_evaluation_result(resultfile: str, metric_name: str, kendall_c, kendall_p, spearman_c, spearman_p, header = None):
    Save individual evaluation results to a CSV file.
- save_evaluation_result_mergemodel(resultfile: str, metric_name: str, metric_evals, header = None):
    Save merged evaluation results over multiple models to a CSV file.
Script Parameters and Task Flags:
---------------------------------
- Dataset-specific paths and number of features (uncomment as needed for different datasets).
- Mode flags: is_classification, is_clustering, is_regression (set to True/False to run evaluation for each task).
- Evaluation type flags: is_evaluate_fsmethod, is_evaluate_metric (set to True/False to enable normalization/evaluation).
Main Evaluation Workflow:
------------------------
- For each task (classification/clustering/regression), for each number of selected features:
    - Loads model accuracies and metric values.
    - Computes and saves correlation (Kendall and Spearman) between them for each model and metric.
    - Optionally normalizes accuracy scores for feature selection methods and saves the normalized results.
Dependencies:
-------------
- numpy
- pandas
- scipy.stats
- os
Usage:
------
Typically executed as a standalone script after setting dataset-specific configurations 
and choosing task/evaluation modes.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import os


# ===================== data loading =====================

def load_clustering_result(file_path = 'result_clustering_num.csv'):
    """
    Loads clustering results from a CSV file and returns them as a NumPy array.
    Parameters:
        file_path (str): Path to the CSV file containing clustering results. 
                         Defaults to 'result_clustering_num.csv'.
    Returns:
        numpy.ndarray: Array of clustering results with values converted to float.
    Notes:
        - The function reads columns 0 to 4 (inclusive) from the CSV file, without headers.
        - Assumes the file exists and is formatted correctly.
    """
    
    usecols=[0,1,2,3,4]
    df = pd.read_csv(file_path, header = None, usecols = usecols, index_col = False)
    accs = df.values.astype(float)
    return accs

def load_classification_result(file_path = 'result_classification_num.csv'):
    """
    Loads classification results from a CSV file and returns them as a NumPy array.
    Parameters:
        file_path (str): Path to the CSV file containing classification results. 
                         Defaults to 'result_classification_num.csv'.
    Returns:
        numpy.ndarray: Array of classification results as floats, extracted from the specified columns of the CSV file.
    """
    usecols=[0,1,2,3,4]
    df = pd.read_csv(file_path, header = None, usecols = usecols, index_col = False)
    
    accs = df.values.astype(float)
    return accs

def load_fsmethod_result(file_path = 'result_fsmethod_num.csv'):
    """
    Loads feature selection method results from a CSV file.
    Parameters:
        file_path (str): Path to the CSV file containing the results. Defaults to 'result_fsmethod_num.csv'.
    Returns:
        numpy.ndarray: Array of accuracy values loaded from the CSV file.
    """
    df = pd.read_csv(file_path, header = 0, index_col = 0)
    accs = df.values.astype(float)
    return accs

def load_metric_result(file_path = 'metric_classification_num.csv'):
    df = pd.read_csv(file_path, header = 0, index_col = False)
    metrics = df.values.astype(float)
    return metrics

def compute_CI(accs, metrics):
    """
    Computes correlation coefficients and p-values between accuracy and FS metrics.
    Parameters:
        accs (list or array-like): List of accuracy values or similar metric.
        metrics (list or array-like): List of metric values to compare with accs.
    Returns:
        tuple: A tuple containing:
            - kendall_c (float): Kendall's tau correlation coefficient.
            - kendall_p (float): Two-sided p-value for Kendall's tau test.
            - spearman_c (float): Spearman's rank correlation coefficient.
            - spearman_p (float): Two-sided p-value for Spearman's rank test.
    """
    
    kendall_c, kendall_p = kendalltau(accs, metrics)
    spearman_c, spearman_p = spearmanr(accs, metrics)
    return kendall_c, kendall_p, spearman_c, spearman_p
def clear_CSV_keepheader(file_path):
    if(os.path.exists(file_path)):
        df_header = pd.read_csv(file_path, nrows=0)
        df_header.to_csv(file_path, index = False)
        
def save_evaluation_result(resultfile, metric_name, kendall_c,kendall_p, spearman_c, spearman_p, header = None):
    pd_new = pd.DataFrame([[metric_name, kendall_c, kendall_p, spearman_c, spearman_p]])
    #pd2 = pd.DataFrame(result_data)
    #pd_new =pd.concat([pd1, pd2], axis = 1)
    if(os.path.exists(resultfile)):
        header = None
    pd_new.to_csv(resultfile, mode='a', header = header, index=False)

def save_evaluation_result_mergemodel(resultfile, metric_name, metric_evals, header = None):
    pd_new = pd.DataFrame(metric_evals.reshape(1,-1), index= [metric_name])
    #pd2 = pd.DataFrame(result_data)
    #pd_new =pd.concat([pd1, pd2], axis = 1)
    if(os.path.exists(resultfile)):
        header = None
    pd_new.to_csv(resultfile, mode='a', header = header)

if __name__ == "__main__":
    ###########working directory###########
    
    ######## Heartdisease dataset #############
    #result_path="/mnt/f/pytorch/feature selection/datasets/Heart Disease/result"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/Heart Disease/metric"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/Heart Disease/result_heartdisease_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/Heart Disease/fsmethod_evalaute_heartdisease"
    #num_features = [2,3,4,6,7,8,10,11,12] #heart disease
    
    ######## Sonar dataset #############
    #result_path="/mnt/f/pytorch/feature selection/datasets/sonar/result"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/sonar/metric"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/sonar/result_sonar_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/sonar/fsmethod_evalaute_sonar"
    #num_features = [6,12,18,24,30,36,42,48,54]#sonar
    
    ######## Spambase dataset #############
    #result_path="/mnt/f/pytorch/feature selection/datasets/spambase/result_spambase"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/spambase/metric_spambase"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/spambase/result_spambase_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/spambase/fsmethod_evalaute_spambase"
    #num_features = [6,12,18,23,29,35,40,46,52]#spanbase
    
    ######## Studentsuccess dataset #############
    #result_path="/mnt/f/pytorch/feature selection/datasets/studentsuccess/result"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/studentsuccess/metric"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/studentsuccess/result_studentsuccess_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/studentsuccess/fsmethod_evalaute_studentsuccess"
    #num_features = [4,8,11,15,18,22,26,29,33]#studentsuccess
   
    ######## Creditcard dataset #############
    #result_path = "/mnt/f/pytorch/feature selection/datasets/creditcard/result_creditcard"
    #metric_path = "/mnt/f/pytorch/feature selection/datasets/creditcard/metric_creditcard"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/creditcard/result_creditcard_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/creditcard/fsmethod_creditcard_evalaute"
    #num_features = [3,5,7,10,12,14,17,19,21]#creditcard

    ######### Automobile dataset ########  
    #result_path="/mnt/f/pytorch/feature selection/datasets/automobile/result_automobile"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/automobile/metric_automobile"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/automobile/result_automobile_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/automobile/fsmethod_automobile_evalaute"
    #num_features = [3,5,8,10,12,15,17,20,22] #automobie
    
    ######### Winequality dataset ########  
    #result_path="/mnt/f/pytorch/feature selection/datasets/winequality/result_winequality"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/winequality/metric_winequality"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/winequality/result_winequality_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/winequality/fsmethod_winequality_evalaute"
    #num_features = [2,3,4,5,6,7,8,9,10] #winequality
    
    ######### Communitiescrime dataset ########  
    #result_path="/mnt/f/pytorch/feature selection/datasets/communitiescrime/result"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/communitiescrime/metric_communitiescrime"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/communitiescrime/result_communitiescrime_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/communitiescrime/fsmethod_communitiescrime_evalaute"
    #num_features = [10,20,30,40,50,60,70,80,90] #Communitiescrime
    
    ######### Waterquality dataset ########  
    #result_path="/mnt/f/pytorch/feature selection/datasets/waterquality/result"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/waterquality/metric_waterquality"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/waterquality/result_waterquality_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/waterquality/fsmethod_waterquality_evalaute"
    #num_features = [2,3,4,5,6,7,8,9,10] #waterquality
    
    ######### Traffic dataset ########
    result_path = "/mnt/f/pytorch/feature selection/datasets/traffic/result"  
    metric_path="/mnt/f/pytorch/feature selection/datasets/traffic/metric_traffic" 
    fs_result_path="/mnt/f/pytorch/feature selection/datasets/traffic/result_fsmethod"
    fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/traffic/fsmethod_traffic_evalaute"
    num_features = [4,8,12,16,20,24,28,32,36] #Traffic
    
    ######### Parkinsons dataset ########
    #result_path="/mnt/f/pytorch/feature selection/datasets/parkinsons/result_parkinsons"
    #metric_path="/mnt/f/pytorch/feature selection/datasets/parkinsons/metric_parkinsons"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/parkinsons/result_parkinsons_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/parkinsons/fsmethod_evalaute_parkinsons"
    #num_features = [3,5,7,9,11,14,16,18,20] #parkinsons
    
    ######## Bankruptcy dataset #############
    #result_path = "/mnt/f/pytorch/feature selection/datasets/bankruptcy/result_bankruptcy"
    #metric_path = "/mnt/f/pytorch/feature selection/datasets/bankruptcy/metric_bankruptcy"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/bankruptcy/result_bankruptcy_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/bankruptcy/fsmethod_bankruptcy_evalaute"
    #num_features = [10,19,28,38,47,56,66,75,84]#bankruptcy
    
    ######## Obesity dataset #############
    #result_path = "/mnt/f/pytorch/feature selection/datasets/obesity/result_obesity"
    #metric_path = "/mnt/f/pytorch/feature selection/datasets/obesity/metric_obesity"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/obesity/result_obesity_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/obesity/fsmethod_obesity_evalaute"
    #num_features = [ 2,  4,  5,  7,  8, 10, 12, 13, 14]#obesity
    
    ######## Yeast dataset #############
    #result_path = "/mnt/f/pytorch/feature selection/datasets/yeast/result_yeast"
    #metric_path = "/mnt/f/pytorch/feature selection/datasets/yeast/metric_yeast"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/yeast/result_yeast_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/yeast/fsmethod_yeast_evalaute"
    #num_features = [ 11,  21,  31,  42,  52,  62,  73,  83,  93]#yeast
    
    ######## Ionosphere dataset #############
    #result_path = "/mnt/f/pytorch/feature selection/datasets/ionosphere/result_ionosphere"
    #metric_path = "/mnt/f/pytorch/feature selection/datasets/ionosphere/metric_ionosphere"
    #fs_result_path="/mnt/f/pytorch/feature selection/datasets/ionosphere/result_ionosphere_fsmethod"
    #fs_evaluate_result_path = "/mnt/f/pytorch/feature selection/datasets/ionosphere/fsmethod_ionosphere_evalaute"
    #num_features = [ 4,  7, 10, 14, 17, 20, 24, 27, 30]#ionosphere
    
    
    clustering_models = ["KMS", "AGC","SPC", "DBC",'BIR']
    classification_models = ["SVM", "RFC", "LRC", "KNN", 'DST']
    regression_models =["LNR", "RDR",'LSR','GBR','GFR']
    metric_names_classification = ['Variance', 'Sim Entropy', 'Rep Entropy', 'SPECtrum','Lap Score', 
                         'Mutual Corre', 'MICI', 'MDCM', 'Linear Depend','MPMR', 'MCFS', 'RSPCA',
                          'ANOVA F', 'ReliefF','Info Gain', 'MRMR', 'ERFS']
    metric_names_clustering = ['Variance', 'Sim Entropy', 'Rep Entropy', 'SPECtrum','Lap Score', 
                         'Mutual Corre', 'MICI', 'MDCM', 'Linear Depend','MPMR', 'MCFS', 'RSPCA']
    
    
    is_classification = False
    is_clustering = False
    is_regression = True
    
    
    is_evaluate_fsmethod = True
    is_evaluate_metric = True
    
    if(is_classification):
        if(is_evaluate_metric):                     
            evaluation_result_file_spearman = f'{result_path}_spearman_eva.csv'
            evaluation_result_file_kendall = f'{result_path}_kendall_eva.csv'
            clear_CSV_keepheader(evaluation_result_file_spearman)
            clear_CSV_keepheader(evaluation_result_file_kendall)
            for num_feature in num_features:
                print(f'computing {num_feature} features...')
                current_accs_file = f'{result_path}_classification_{num_feature}.csv'
                current_metrics_file = f'{metric_path}_classification_{num_feature}.csv'
                accs = load_clustering_result(current_accs_file)
                metrics = load_metric_result(current_metrics_file)
                metric_id = 0
                
                for metric_name in metric_names_classification:
                    cur_metric_values = metrics[:,metric_id]
                    model_accs_id = 0
                    metric_id += 1
                    kendall_cs = np.zeros((len(classification_models)))
                    spearman_cs = np.zeros((len(classification_models)))
                    for classificaion_model in classification_models:
                        curr_model_accs = accs[:,model_accs_id]
                        kendall_c, kendall_p, spearman_c, spearman_p = np.abs(compute_CI(curr_model_accs, cur_metric_values))
                        kendall_cs[model_accs_id] = kendall_c
                        spearman_cs[model_accs_id] = spearman_c
                        model_accs_id += 1
                    save_evaluation_result_mergemodel(evaluation_result_file_spearman,metric_name,spearman_cs,header = classification_models)
                    save_evaluation_result_mergemodel(evaluation_result_file_kendall,metric_name,kendall_cs,header = classification_models) 
        if(is_evaluate_fsmethod):            
            for num_feature in num_features:
                print(f'computing {num_feature} features...')
                current_accs_file = f'{result_path}_classification_{num_feature}.csv'
                current_fsresult_file = f'{fs_result_path}_{num_feature}.csv'
                cur_fs_evaluate_file = f'{fs_evaluate_result_path}_{num_feature}.csv'
                accs = load_classification_result(current_accs_file)#50 rows * 5 cols
                df = pd.read_csv(current_fsresult_file, index_col = 0)#18 rows * 5 cols,
                row_labels = df.index
                col_labels = df.columns           
                fs_accs = df.values
                
                for col in range(accs.shape[1]):
                    cur_min = accs[:,col].min()
                    cur_max = accs[:,col].max()
                    fs_accs[:,col] = (fs_accs[:,col]-cur_min)/(cur_max - cur_min + 1e-8)
                    fs_accs[:,col] = np.clip(fs_accs[:,col], 0, 1)
                pd_fs_evalaute_result =  pd.DataFrame(fs_accs,
                                                index = row_labels,
                                                columns = col_labels)
                pd_fs_evalaute_result.to_csv(cur_fs_evaluate_file)
            
    if(is_clustering): #clustering
        if(is_evaluate_metric):
            evaluation_result_file_spearman = f'{result_path}_spearman_eva.csv'
            evaluation_result_file_kendall = f'{result_path}_kendall_eva.csv'
            clear_CSV_keepheader(evaluation_result_file_spearman)
            clear_CSV_keepheader(evaluation_result_file_kendall)
            for num_feature in num_features:
                print(f'computing {num_feature} features...')
                current_accs_file = f'{result_path}_clustering_{num_feature}.csv'
                current_metrics_file = f'{metric_path}_clustering_{num_feature}.csv'
                accs = load_clustering_result(current_accs_file)
                metrics = load_metric_result(current_metrics_file)
                metric_id = 0
                
                for metric_name in metric_names_clustering:
                    cur_metric_values = metrics[:,metric_id]
                    model_accs_id = 0
                    metric_id += 1
                    kendall_cs = np.zeros((len(clustering_models)))
                    spearman_cs = np.zeros((len(clustering_models)))
                    for clustering_model in clustering_models:
                        curr_model_accs = accs[:,model_accs_id]
                        kendall_c, kendall_p, spearman_c, spearman_p = np.abs(compute_CI(curr_model_accs, cur_metric_values))
                        kendall_cs[model_accs_id] = kendall_c
                        spearman_cs[model_accs_id] = spearman_c
                        model_accs_id += 1
                    save_evaluation_result_mergemodel(evaluation_result_file_spearman,metric_name,spearman_cs,header = clustering_models)
                    save_evaluation_result_mergemodel(evaluation_result_file_kendall,metric_name,kendall_cs,header = clustering_models) 
        if(is_evaluate_fsmethod):            
            for num_feature in num_features:
                print(f'computing {num_feature} features...')
                current_accs_file = f'{result_path}_clustering_{num_feature}.csv'
                current_fsresult_file = f'{fs_result_path}_{num_feature}.csv'
                cur_fs_evaluate_file = f'{fs_evaluate_result_path}_{num_feature}.csv'
                accs = load_classification_result(current_accs_file)#50 rows * 5 cols
                df = pd.read_csv(current_fsresult_file, index_col = 0)#18 rows * 5 cols,含标签
                row_labels = df.index
                col_labels = df.columns           
                fs_accs = df.values
                
                for col in range(accs.shape[1]):
                    cur_min = accs[:,col].min()
                    cur_max = accs[:,col].max()
                    fs_accs[:,col] = (fs_accs[:,col]-cur_min)/(cur_max - cur_min + 1e-8)
                    fs_accs[:,col] = np.clip(fs_accs[:,col], 0, 1)
                pd_fs_evalaute_result =  pd.DataFrame(fs_accs,
                                                index = row_labels,
                                                columns = col_labels)
                pd_fs_evalaute_result.to_csv(cur_fs_evaluate_file)   

    if(is_regression):
        if(is_evaluate_metric):                     
            evaluation_result_file_spearman = f'{result_path}_spearman_eva.csv'
            evaluation_result_file_kendall = f'{result_path}_kendall_eva.csv'
            clear_CSV_keepheader(evaluation_result_file_spearman)
            clear_CSV_keepheader(evaluation_result_file_kendall)
            for num_feature in num_features:
                print(f'computing {num_feature} features...')
                current_accs_file = f'{result_path}_regression_{num_feature}.csv'
                current_metrics_file = f'{metric_path}_regression_{num_feature}.csv'
                accs = load_clustering_result(current_accs_file)
                metrics = load_metric_result(current_metrics_file)
                metric_id = 0
                
                for metric_name in metric_names_classification:
                    cur_metric_values = metrics[:,metric_id]
                    model_accs_id = 0
                    metric_id += 1
                    kendall_cs = np.zeros((len(regression_models)))
                    spearman_cs = np.zeros((len(regression_models)))
                    for regression_model in regression_models:
                        curr_model_accs = accs[:,model_accs_id]
                        kendall_c, kendall_p, spearman_c, spearman_p = np.abs(compute_CI(curr_model_accs, cur_metric_values))
                        kendall_cs[model_accs_id] = kendall_c
                        spearman_cs[model_accs_id] = spearman_c
                        model_accs_id += 1
                    save_evaluation_result_mergemodel(evaluation_result_file_spearman,metric_name,spearman_cs,header = regression_models)
                    save_evaluation_result_mergemodel(evaluation_result_file_kendall,metric_name,kendall_cs,header = regression_models) 
        if(is_evaluate_fsmethod):            
            for num_feature in num_features:
                print(f'computing {num_feature} features...')
                current_accs_file = f'{result_path}_regression_{num_feature}.csv'
                current_fsresult_file = f'{fs_result_path}_{num_feature}.csv'
                cur_fs_evaluate_file = f'{fs_evaluate_result_path}_{num_feature}.csv'
                accs = load_classification_result(current_accs_file)#50 rows * 5 cols
                df = pd.read_csv(current_fsresult_file, index_col = 0)#18 rows * 5 cols,含标签
                row_labels = df.index
                col_labels = df.columns           
                fs_accs = df.values
                
                for col in range(accs.shape[1]):
                    cur_min = accs[:,col].min()
                    cur_max = accs[:,col].max()
                    fs_accs[:,col] = (fs_accs[:,col]-cur_min)/(cur_max - cur_min + 1e-8)
                    fs_accs[:,col] = np.clip(fs_accs[:,col], 0, 1)
                pd_fs_evalaute_result =  pd.DataFrame(fs_accs,
                                                index = row_labels,
                                                columns = col_labels)
                pd_fs_evalaute_result.to_csv(cur_fs_evaluate_file)