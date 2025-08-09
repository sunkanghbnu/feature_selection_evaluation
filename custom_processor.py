# featuremetrics.py
'''
This module provides utility functions for managing feature selection results, metrics, and combinations using CSV files. 
It includes functions to save, clear, and check feature combinations and metrics, as well as to load combinations from files.
Functions
---------
SaveResult(filename, feature_indices, accuracy, feature_scores)
    Save feature selection results (accuracy, feature scores, and feature indices) to a CSV file.
clear_CSV_keepheader(file_path)
    Clear the contents of a CSV file while preserving the header row.
get_featureconbinations_from_csv(csvfilename, num_features, num_accurices_cols=1)
    Load feature combinations from a CSV file, extracting feature indices after accuracy columns.
check_featurecombination_exist(existed_featurecombinations, new_featurecombination)
    Check if a given feature combination already exists within a collection of combinations.
save_featurecombiantion_result(csvfilename, accuracy, feature_combination)
    Save a single feature combination and its accuracy to a CSV file.
save_metric_result(csvfilename, metrics, header=None)
    Save metric results to a CSV file, optionally specifying column headers.
get_num_conputed_metrics_from_csv(csvfilename)
    Return the number of computed metric records in a given CSV file.
'''
import numpy as np
import pandas as pd
import math, os


def SaveResult(filename, feature_indices, accuracy, feature_scores):
    """
    Save feature selection results to a CSV file.
    
    Parameters:
    filename : str
        Path to the output CSV file.
    feature_indices : ndarray
        Array of selected feature indices.
    accuracy : float or ndarray
        Accuracy metric(s) of the feature selection.
    feature_scores : float or ndarray
        Score(s) of the selected features.
    """
    df3=pd.DataFrame(feature_indices.reshape(1,-1))
    df1=pd.DataFrame(np.array(accuracy).reshape(1,-1))
    df2=pd.DataFrame(np.array(feature_scores).reshape(1,-1))
    df_new=pd.concat([df1,df2,df3],axis=1)
    df_new.to_csv(filename, mode='a', header=False, index=False) 

def clear_CSV_keepheader(file_path):
    """
    Clear CSV file content while preserving the header row.
    
    Parameters:
    file_path : str
        Path to the CSV file to be cleared.
    """
    df_header = pd.read_csv(file_path, nrows=0)
    df_header.to_csv(file_path, index=False)
    

def get_featureconbinations_from_csv(csvfilename:str, num_features:int, num_accurices_cols = 1)->np.ndarray:
    """
    Load feature combinations from CSV file.
    
    Parameters:
    csvfilename : str
        Path to CSV file containing feature combinations.
    num_features : int
        Expected number of features per combination.
    num_accuracy_cols : int, optional
        Number of accuracy columns preceding feature indices.
    
    Returns:
    feature_combinations : ndarray
        Array of feature index combinations.
    """
    df = pd.read_csv(csvfilename, header = None)
    feature_combinations = df.iloc[:, num_accurices_cols:].values.astype(int)
    assert feature_combinations.shape[1] == num_features, (
        f"Expected {num_features} features, found {feature_combinations.shape[1]}"
    )
    if(feature_combinations.ndim == 1 ):
        np.expand_dims(feature_combinations,axis = 0)
    return feature_combinations

def check_featurecombination_exist(existed_featurecombinations:np.ndarray, new_featurecombination:np.ndarray)->bool:
    """
    Check if feature combination exists in collection.
    
    Parameters:
    existing_combinations : ndarray
        Array of existing feature combinations.
    new_combination : ndarray
        New feature combination to check.
    
    Returns:
    exists : bool
        True if combination exists, False otherwise.
    """
    for row in range(existed_featurecombinations.shape[0]):
        if (np.array_equal(existed_featurecombinations[row,:], new_featurecombination)):
            return True
    return False
    #matches = (existed_featurecombinations == new_featurecombination).all(axis=0)
    #return np.any(matches)

def save_featurecombiantion_result(csvfilename:str, accuracy:np.ndarray, feature_combination:np.ndarray):
    """
    Save feature combination result to CSV.
    
    Parameters:
    csvfilename : str
        Output CSV file path.
    accuracy : float
        Accuracy metric for the combination.
    feature_combination : ndarray
        Array of feature indices.
    """
    df1=pd.DataFrame(np.array(accuracy).reshape(1,-1))
    df2=pd.DataFrame(feature_combination.reshape(1,-1))
    df_new=pd.concat([df1,df2], axis = 1)  
    df_new.to_csv(csvfilename, mode='a', header = None, index = False)

def save_metric_result(csvfilename:str, metrics:np.ndarray, header = None):
    """
    Save metric results to CSV file.
    
    Parameters:
    csvfilename : str
        Output CSV file path.
    metrics : ndarray
        Array of metric values to save.
    header : list, optional
        Column headers for the CSV file.
    """
    df1=pd.DataFrame(np.array(metrics).reshape(1,-1))
    if os.path.exists(csvfilename):
        header = None
    df1.to_csv(csvfilename, mode='a',
                  header = header, 
                  index = False)
def get_num_conputed_metrics_from_csv(csvfilename:str)->int:
    """
    Get number of computed metrics from CSV file.
    
    Parameters:
    csvfilename : str
        Path to CSV file containing metrics.
    
    Returns:
    num_metrics : int
        Number of metric records in the file.
    """
    df = pd.read_csv(csvfilename, header = 0)
    return df.shape[0]