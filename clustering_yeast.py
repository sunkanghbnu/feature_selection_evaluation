#clustering_yeast.py
import numpy as np
import pandas as pd
import os,sys
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering,DBSCAN,AffinityPropagation,Birch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils_accuarcy_metric_compute import *
from hdbscan import HDBSCAN
import warnings
#from warnings import FutureWarning

# 屏蔽特定警告
warnings.filterwarnings("ignore", message=".*'force_all_finite'.*") 

######### data loading and preprocessing #########
def load_yeast_data(file_path):
    data, meta = arff.loadarff(file_path)  # [1,6](@ref)
    df = pd.DataFrame(data)

    features = df.iloc[:, :-14]
    labels = df.iloc[:, -14:].applymap(lambda x: 1 if x == b'1' else 0)  # transform byte strings to integers
       
    # fill missing values
    features = features.fillna(features.mean()) 
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features) 
    
    multilabels = df.iloc[:, -14:].values.astype(int)
    y_true = np.argmax(multilabels, axis=1)
   
    return features_scaled, labels.values, y_true, features



if __name__ == "__main__":
    # 数据集路径（需替换为实际路径）
    file_path_yeast = "/mnt/f/pytorch/feature selection/datasets/yeast/yeast.arff" 
    result_path ="/mnt/f/pytorch/feature selection/datasets/yeast/result_yeast"
    metric_path = "/mnt/f/pytorch/feature selection/datasets/yeast/metric_yeast"
    
    X, multi_lbl_true, y_true, X_org= load_yeast_data(file_path_yeast)
    print(f"数据集形状: {X.shape}, 标签形状: {multi_lbl_true.shape}, {y_true.shape}")
    num_samples, num_total_features = X.shape
    num_clusters = 14
    
    models = {
        "knn_clustering": KMeans(n_clusters = num_clusters, random_state = 42, n_init = 10),
        "agg_clustering": AgglomerativeClustering(n_clusters = num_clusters, linkage='ward'),
        "spec_clustering": SpectralClustering(n_clusters = num_clusters, affinity = 'nearest_neighbors', n_neighbors = 10, assign_labels = 'kmeans', random_state = 42),
        "dbscan_clustering": HDBSCAN(min_cluster_size = 6, min_samples = 3),
        'brich_clustering': Birch(n_clusters = num_clusters)
        }
    
    #模式开关
    is_compute_accuracy = False
    is_test_fs_method = False
    is_compute_metric = False
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
                                            models=models, result_path=result_path, is_multiple_labels=True,multi_lbl_true=multi_lbl_true,
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
                                                                          result_path=result_path, is_multiple_labels=True,multi_lbl_true=multi_lbl_true)