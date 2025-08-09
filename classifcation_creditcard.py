# classification/classifcation_creditcard.py
import numpy as np
import pandas as pd
import os,sys,math, time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils_accuarcy_metric_compute import *

# ========== Data loading and preprocessing ========== #
def load_creditcard_data(file_path='creditcard.csv'):
    
    df = pd.read_csv(file_path)
    # data preprocessing
    df.replace('?', np.nan, inplace=True)
    df = df.dropna(axis = 0)  # delete rows with missing values
    X = df.iloc[:, :-1].values.astype(int)
    y_ture = df['default payment next month'].values# transform to int

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_ture, X

    
if __name__ == "__main__":
    ########### Working directory in WSL ############
    file_path="/mnt/f/pytorch/feature selection/datasets/creditcard/creditcard.csv"
    result_path = "/mnt/f/pytorch/feature selection/datasets/creditcard/result_creditcard"
    metric_path = "/mnt/f/pytorch/feature selection/datasets/creditcard/metric_creditcard"
    X, y_true, X_org  = load_creditcard_data(file_path)
    print(f"data size: {X.shape}, label size: {y_true.shape}")
    
    num_samples, num_total_features = X.shape

    is_compute_accuracy = True
    is_test_fs_method = True
    is_compute_metric = True
    is_overwrite_metric = True
    
    num_steps = 10
    num_sets = np.ceil(np.linspace(0.1, 1.0, num_steps)*num_total_features).astype(int)
    print(f'num_sets = {num_sets}')
    
    models = {
        "svm_classification": SVC(C = 0.1, kernel = 'rbf', gamma = 0.05, random_state = 42),    
        "rft_classification": RandomForestClassifier(n_estimators = 100, random_state = 42),
        "lgn_classification": LogisticRegression(solver = 'liblinear', max_iter = 1000),
        "knn_classification": KNeighborsClassifier(n_neighbors = 5),
        'dtr_classification': DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 42)
        }
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size = 0.7, random_state = 42)   
    
    ################ Random feature selection and classification #################
    new_iter_times = 50
    max_iter_times = 500
    
    ############# random feature selection and classification accuracy ##########
    if is_compute_accuracy:
        random_feature_selection_classification(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,
                                                num_sets= num_sets, num_total_features=num_total_features,
                                                models=models, result_path=result_path, new_iter_times=new_iter_times,max_iter_times=max_iter_times)
    ############# feature metrics computation for all feature subsets ##########
    if is_compute_metric:
        compute_metrics_for_combinations_for_classification(X_metric_use=X_test, y_metric_use=y_test,X_org = X_org,
                                                            num_sets=num_sets, num_total_features=num_total_features,
                                                            result_path=result_path, metric_path=metric_path,
                                                            is_overwrite_metric=is_overwrite_metric)
    ############# feature selection by 17 methods ##########
    if is_test_fs_method:
        fs_methods_evaluation_for_classification(X_fsmethod_use=X_train, X_test=X_test, X_org=X_org, y_fsmethod_use=y_train, y_test=y_test,
                                                 num_sets=num_sets, num_total_features=num_total_features,
                                                 models=models, result_path=result_path)