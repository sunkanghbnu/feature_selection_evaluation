# -*- coding: utf-8 -*-
"""
Wine Quality 数据集回归分析完整实现
数据集：1599个红酒样本（11个化学属性 + 品质评分）
作者：DeepSeek
日期：2025-06-07
"""
import numpy as np
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils_accuarcy_metric_compute import *

######### data loading and preprocessing #########

def load_communitiescrime_data(file_path=''):
    df = pd.read_csv(file_path, header = None, index_col = None, na_values=["?"])  # 
    missing_threshold = 0.3  # drop columns with more than 30% missing values
    df = df.loc[:, df.isnull().mean(axis = 0) < missing_threshold]
    print(f'df2= {df.shape}')
    df.fillna(df.median(), inplace = True)
    X = df.iloc[:, :-1].values.astype(float)
    y_true = df.iloc[:, -1].values.astype(float)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_true, X
    
if __name__ == "__main__":
    ######### working directory in WSL #########
    file_path="/mnt/f/pytorch/feature selection/datasets/communitiescrime/communities.csv"  
    result_path="/mnt/f/pytorch/feature selection/datasets/communitiescrime/result"
    metric_path = "/mnt/f/pytorch/feature selection/datasets/communitiescrime/metric_communitiescrime"
    X,  y_true, X_org = load_communitiescrime_data(file_path)
    print(f"data size: {X.shape}, label size: {y_true.shape}")

    num_samples, num_total_features = X.shape

    is_compute_accuracy = True
    is_test_fs_method = True
    is_compute_metric = True
    is_overwrite_metric = True
    
    num_steps = 10
    num_sets = np.ceil((np.linspace(0.1, 1.0, num_steps) - 1e-8)*num_total_features).astype(int)
    print(f'num_sets = {num_sets}')
    
    models = {
        "LinearRegression": LinearRegression(n_jobs = 20),
        "RidgeRegression": Ridge(alpha = 1.0, random_state = 42), 
        "LassoRegression": Lasso(alpha = 0, random_state = 42),
        "GradientBoostingRegression": GradientBoostingRegressor(n_estimators = 100, random_state = 42),
        "RandomForestRegression": RandomForestRegressor(n_estimators = 100, random_state = 42)
    }      
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size = 0.6, random_state = 42)
    print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')

    
    ######### compute accuracy for random feature combinations #########
    new_iter_times = 50
    max_iter_times = 500
    ############# random feature selection and regression accuracy ##########
    if is_compute_accuracy:
        random_feature_selection_regression(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test,
                                            num_sets = num_sets, num_total_features = num_total_features,
                                            models = models, result_path = result_path, 
                                            new_iter_times = new_iter_times, max_iter_times = max_iter_times)
    
    ######### compute metrics for random feature combinations #########           
    if is_compute_metric:
        compute_metrics_for_combinations_for_regression(X_metric_use = X_test, X_org=X_org,y_metric_use = y_test,
                                                        num_sets = num_sets, num_total_features = num_total_features,
                                                        result_path = result_path, metric_path = metric_path,
                                                        is_overwrite_metric = is_overwrite_metric)
    
    ######### test feature selection methods #########
    if is_test_fs_method:
        fs_method_selected_results = fs_methods_evaluation_for_regression(X_fsmethod_use = X_train, X_test = X_test,X_org=X_org,
                                                                          y_fsmethod_use = y_train, y_test = y_test,
                                                                          num_sets = num_sets, num_total_features = num_total_features,
                                                                          models = models, result_path = result_path)