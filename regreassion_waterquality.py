# File: regression/regression_waterquality.py
import numpy as np
import pandas as pd
import os, sys
import scipy.io as sio
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils_accuarcy_metric_compute import *


######### data loading and preprocessing #########
def load_waterquality_data(file_path = 'data.mat'):
    
    data = sio.loadmat(file_path, squeeze_me = True)
    X_train_full = data["X_tr"]
    X_test_full = data["X_te"]
    y_train_full = data['Y_tr']
    y_test_full = data['Y_te']
    
    n_samples = 423
    n_features = 11
    location_id = 5 # max = 36
    X_train = np.zeros((n_samples,n_features))
    X_test =np.zeros((282, n_features))
    y_train = y_train_full[location_id]
    y_test = y_test_full[location_id]
    for i in range(n_samples):
        X_train[i,:] = X_train_full[i][location_id,:]
    for i in range(282):
        X_test[i,:] = X_test_full[i][location_id,:]

    return X_train, y_train, X_test, y_test
    
if __name__ == "__main__":
    ######### working directory #########
    file_path="/mnt/f/pytorch/feature selection/datasets/waterquality/water_dataset.mat"  
    result_path="/mnt/f/pytorch/feature selection/datasets/waterquality/result"
    metric_path = "/mnt/f/pytorch/feature selection/datasets/waterquality/metric_waterquality"
    
    X_train,  y_train, X_test, y_test = load_waterquality_data(file_path)
    print(f"train data size: {X_train.shape}, test data size: {X_test.shape}")
    num_samples, num_total_features = X_train.shape
   
    is_compute_accuracy = True
    is_test_fs_method = True
    is_compute_metric = True
    is_overwrite_metric = True
    
    num_steps = 10
    num_sets = np.ceil((np.linspace(0.1, 1.0, num_steps) - 1e-8)*num_total_features).astype(int)
    print(f'num_sets = {num_sets}')
    
    models = {
        "LinearRegression": LinearRegression(n_jobs = 20),
        "RidgeRegression": Ridge(alpha = 0.1, random_state = 42), 
        "LassoRegression": Lasso(alpha = 0, random_state = 42),
        "GradientBoostingRegression": GradientBoostingRegressor(n_estimators = 100, random_state = 42),
        "RandomForestRegression": RandomForestRegressor(n_estimators = 100, random_state = 42)
    }        
    
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
        compute_metrics_for_combinations_for_regression(X_metric_use = X_test, y_metric_use = y_test, X_org = X_test, 
                                                        num_sets = num_sets, num_total_features = num_total_features,
                                                        result_path = result_path, metric_path = metric_path,
                                                        is_overwrite_metric = is_overwrite_metric)
    
    ######### test feature selection methods #########
    if is_test_fs_method:
        fs_method_selected_results = fs_methods_evaluation_for_regression(X_fsmethod_use = X_train, X_test = X_test, X_org= X_train,
                                                                          y_fsmethod_use = y_train, y_test = y_test,
                                                                          num_sets = num_sets, num_total_features = num_total_features,
                                                                          models = models, result_path = result_path)