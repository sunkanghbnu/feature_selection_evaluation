# ========== 1. 环境准备 ==========
import numpy as np
import pandas as pd

import os,sys,copy,math,time
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering,DBSCAN,AffinityPropagation,Birch
from hdbscan import HDBSCAN

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils_accuarcy_metric_compute import *
import warnings
#from warnings import FutureWarning

# 屏蔽特定警告
warnings.filterwarnings("ignore", message=".*'force_all_finite'.*") 


# ========== data loading and preprocessing ==============
def load_bankruptcy_data(file_path='bankruptcy.csv'):

    df = pd.read_csv(file_path)


    X = df.iloc[:, 1:].values.astype(float)
    y_ture = df.iloc[:, 0].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_ture, X


if __name__ == "__main__":
    ########## working directory #############
    file_path="/mnt/f/pytorch/feature selection/datasets/bankruptcy/bankruptcy.csv"
    result_path = "/mnt/f/pytorch/feature selection/datasets/bankruptcy/result_bankruptcy"
    metric_path = "/mnt/f/pytorch/feature selection/datasets/bankruptcy/metric_bankruptcy"
    X, y_true, X_org  = load_bankruptcy_data(file_path)

    print(f"data size: {X.shape}, label size: {y_true.shape}")
    
    num_samples, num_total_features = X.shape
    num_clusters = 2
    
    models = {
        "knn_clustering": KMeans(n_clusters = num_clusters, random_state = 42, n_init = 10),
        "agg_clustering": AgglomerativeClustering(n_clusters = num_clusters, linkage='ward'),
        "spec_clustering": SpectralClustering(n_clusters = num_clusters, affinity = 'nearest_neighbors', n_neighbors = 10, assign_labels = 'kmeans', random_state = 42, n_jobs=20),
        "dbscan_clustering": HDBSCAN(min_cluster_size = 20, min_samples = 10),
        #'affPro_clustering': AffinityPropagation(damping = 0.9, max_iter = 1000, random_state = 42),
        'brich_clustering': Birch(n_clusters = num_clusters)
        }
    
    #模式开关
    is_compute_accuracy = True
    is_test_fs_method = True
    is_compute_metric = True
    is_overwrite_metric = True

    num_steps = 10
    num_sets = np.ceil((np.linspace(0.1, 1.0, num_steps) - 1e-8)*num_total_features).astype(int)
    print(f'num_sets = {num_sets}')
    
    ################ Random feature selection and clustering #################
    new_iter_times = 50
    max_iter_times = 500
    
    ############# random feature selection and clustering accuracy ##########
    if is_compute_accuracy:
        random_feature_selection_clustering(X=X, y_true=y_true, num_sets=num_sets, num_total_features=num_total_features,
                                            models=models, result_path=result_path, is_multiple_labels=False,multi_lbl_true=None,
                                            new_iter_times=new_iter_times, max_iter_times=max_iter_times)
    ############# feature metrics computation for all feature subsets ##########            
    if is_compute_metric:
        compute_metrics_for_combinations_for_clustering(X = X, X_org=X_org, num_sets=num_sets, num_total_features=num_total_features,
                                                        num_clusters= num_clusters, result_path=result_path, metric_path= metric_path,
                                                        is_overwrite_metric=is_overwrite_metric)
        
    ############# feature selection by 12 methods ##########
    if is_test_fs_method:
        fs_method_selected_results = fs_methods_evaluation_for_clustering(X=X, X_org=X_org, y_true=y_true,
                                                                          num_sets=num_sets, num_total_features=num_total_features,
                                                                          models=models, num_clusters=num_clusters,
                                                                          result_path=result_path, is_multiple_labels=False,multi_lbl_true=None)