# ========== 1. 环境准备 ==========
import numpy as np
import pandas as pd
import os,sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering,DBSCAN,Birch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils_accuarcy_metric_compute import *
from hdbscan import HDBSCAN

import warnings
warnings.filterwarnings("ignore", message=".*'force_all_finite'.*") 


# ========== 2. 数据加载与预处理 ==========
def load_parkinsons_data(file_path):
    """加载并预处理Wine数据集"""
    
    df = pd.read_csv(file_path, index_col = 0, header = 0)
    X = df.iloc[:,:-1].values.astype(float)# 特征矩阵
    y_true = df['status'].values.astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_true,X


if __name__ == "__main__":
    # 数据集路径（需替换为实际路径）
    #file_path_yeast = "/mnt/f/pytorch/feature selection/datasets/yeast/yeast.arff"  # 从UCI下载：https://archive.ics.uci.edu/ml/datasets/Yeast
    file_path = "/mnt/f/pytorch/feature selection/datasets/parkinsons/parkinsons.csv"
    result_path = "/mnt/f/pytorch/feature selection/datasets/parkinsons/result_parkinsons"
    metric_path = "/mnt/f/pytorch/feature selection/datasets/parkinsons/metric_parkinsons"
    X, y_true, X_org= load_parkinsons_data(file_path)
    print(f"数据集形状: {X.shape}, 标签形状: {y_true.shape}")
    
    num_samples, num_total_features = X.shape
    num_clusters = 2
    
    models = {
        "knn_clustering": KMeans(n_clusters = num_clusters, random_state = 42, n_init = 10),
        "agg_clustering": AgglomerativeClustering(n_clusters = num_clusters, linkage='ward'),
        "spec_clustering": SpectralClustering(n_clusters = num_clusters, affinity = 'nearest_neighbors', n_neighbors = 10, assign_labels = 'kmeans', random_state = 42),
        "dbscan_clustering": HDBSCAN(min_cluster_size = 20, min_samples = 10),
        'birch_clustering': Birch(n_clusters = num_clusters)
        }
    
    #模式开关
    is_compute_accuracy = False
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