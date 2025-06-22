import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_submission_metrics(submission_file, ground_truth_file, primary_key_candidates=['Id', 'ID']):
    """
    Compute Accuracy and AUC for a submission CSV file using flexible primary key.
    
    Parameters:
    submission_file (str): Path to submission CSV file with columns ['Id' or 'ID', 'class']
    ground_truth_file (str): Path to ground truth CSV file with columns ['Id' or 'ID', 'class']
    primary_key_candidates (list): List of possible primary key column names
    
    Returns:
    dict: Dictionary containing 'accuracy' and 'auc' scores
    """
    try:
        # Loading submission and ground truth data
        submission_df = pd.read_csv(submission_file)
        ground_truth_df = pd.read_csv(ground_truth_file)
        
        # Determine the primary key
        primary_key = None
        for key in primary_key_candidates:
            if key in submission_df.columns and key in ground_truth_df.columns:
                primary_key = key
                break
        if primary_key is None:
            raise ValueError("No valid primary key ('Id' or 'ID') found in both files")
        
        # Validating required columns
        if 'class' not in submission_df.columns:
            raise ValueError("Submission file must contain 'class' column")
        if 'class' not in ground_truth_df.columns:
            raise ValueError("Ground truth file must contain 'class' column")
        
        # Merging dataframes on primary_key to align predictions and ground truth
        merged_df = pd.merge(
            submission_df[[primary_key, 'class']],
            ground_truth_df[[primary_key, 'class']],
            on=primary_key,
            suffixes=('_pred', '_true')
        )
        
        # Checking for missing matches
        if merged_df.empty:
            raise ValueError("No matching IDs found between submission and ground truth")
        
        y_true = merged_df['class_true'].astype(int)
        y_pred = merged_df['class_pred'].astype(int)
        
        # Computing Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Computing AUC (multi-class, one-vs-rest)
        classes = np.unique(y_true)
        if len(classes) < 2:
            raise ValueError("At least two classes are required for AUC computation")
        
        # Binarizing labels for multi-class ROC AUC
        y_true_bin = label_binarize(y_true, classes=classes)
        y_pred_bin = label_binarize(y_pred, classes=classes)
        
        # Calculating AUC using one-vs-rest and macro averaging
        auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovr')
        
        return {
            'accuracy': round(accuracy, 4),
            'auc': round(auc, 4),
        }
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error computing metrics: {str(e)}")

def compute_multilabel_metrics(submission_file, ground_truth_file, primary_key_candidates=['Id', 'ID']):
    """
    Compute per-class and macro-averaged Accuracy and AUC for a multi-label classification submission CSV file.
    
    Parameters:
    submission_file (str): Path to submission CSV file with columns ['Id' or 'ID', 'class_1', 'class_2', ...]
    ground_truth_file (str): Path to ground truth CSV file with columns ['Id' or 'ID', 'class_1', 'class_2', ...]
    primary_key_candidates (list): List of possible primary key column names
    
    Returns:
    dict: Dictionary containing:
        - 'per_class_accuracy': Dict of accuracy scores per class
        - 'macro_accuracy': Macro-averaged accuracy across classes
        - 'per_class_auc': Dict of AUC scores per class
        - 'macro_auc': Macro-averaged AUC across classes
    """
    try:
        # Loading submission and ground truth data
        submission_df = pd.read_csv(submission_file)
        ground_truth_df = pd.read_csv(ground_truth_file)
        
        # Determine the primary key
        primary_key = None
        for key in primary_key_candidates:
            if key in submission_df.columns and key in ground_truth_df.columns:
                primary_key = key
                break
        if primary_key is None:
            raise ValueError("No valid primary key ('Id' or 'ID') found in both files")
        
        # Identifying class columns (columns starting with 'class_')
        class_columns = [col for col in submission_df.columns if col.startswith('class_')]
        if not class_columns:
            raise ValueError("Submission file must contain at least one 'class_' column")
        
        # Verify class columns match in both files
        gt_class_columns = [col for col in ground_truth_df.columns if col.startswith('class_')]
        if sorted(class_columns) != sorted(gt_class_columns):
            raise ValueError("Class columns in submission and ground truth files do not match")
        
        # Merging dataframes on primary_key to align predictions and ground truth
        merged_df = pd.merge(
            submission_df[[primary_key] + class_columns],
            ground_truth_df[[primary_key] + class_columns],
            on=primary_key,
            suffixes=('_pred', '_true')
        )
        
        # Checking for missing matches
        if merged_df.empty:
            raise ValueError("No matching IDs found between submission and ground truth")
        
        # Initialize dictionaries to store per-class metrics
        per_class_accuracy = {}
        per_class_auc = {}
        accuracy_scores = []
        auc_scores = []
        
        # Compute metrics for each class
        for class_col in class_columns:
            # Extract true and predicted labels for the class
            y_true = merged_df[f'{class_col}_true'].astype(int)
            y_pred = merged_df[f'{class_col}_pred'].astype(int)
            
            # Compute accuracy for the class
            accuracy = accuracy_score(y_true, y_pred)
            per_class_accuracy[class_col] = round(accuracy, 4)
            accuracy_scores.append(accuracy)
            
            # Compute AUC for the class
            # Ensure there are at least two classes (0 and 1) for AUC
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_pred)
                per_class_auc[class_col] = round(auc, 4)
                auc_scores.append(auc)
            else:
                per_class_auc[class_col] = None  # AUC not defined for single class
                
        # Compute macro-averaged metrics
        macro_accuracy = round(np.mean(accuracy_scores), 4)
        macro_auc = round(np.mean([auc for auc in auc_scores if auc is not None]), 4) if auc_scores else None
        
        return {
            'per_class_accuracy': per_class_accuracy,
            'macro_accuracy': macro_accuracy,
            'accuracy': macro_accuracy,
            'per_class_auc': per_class_auc,
            'macro_auc': macro_auc,
            'auc': macro_auc,
        }
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error computing metrics: {str(e)}")

def compute_all_metrics(dataset_gt_path, experiment_list):
    """
    Compute metrics for all experiments and save results to a CSV file.
    
    Parameters:
    dataset_gt_path (str): Path to the ground truth CSV file
    experiment_list (list): List of experiment directory names
    output_csv (str): Path to save the output CSV file
    """
    results = []
    
    for exp in experiment_list:
        exp_pred_path = f"logs/{exp}/submission.csv"
        try:
            if "chest" in exp_pred_path:
                metrics = compute_multilabel_metrics(exp_pred_path, dataset_gt_path)
            else:
                metrics = compute_submission_metrics(exp_pred_path, dataset_gt_path)
            print(f"[{exp}] ACC: {metrics['accuracy']:.4f}")
            print(f"[{exp}] AUC: {metrics['auc']:.4f}")
            results.append({
                'Experiment': exp,
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['auc'],
                'Error': None
            })
        except Exception as e:
            print(f"[{exp}] Error: {str(e)}")
            results.append({
                'Experiment': exp,
                'Accuracy': None,
                'AUC': None,
                'Error': str(e)
            })

from scripts.config_to_be_eval import *

### example config:
# automl_config = [
#     {
#         'dataset': 'Automl_RetinaMNIST',
#         'dataset_gt_path': 'meddle/med_dl_tasks/retinamnist_224/test.csv',
#         'experiment_list': [
#             "92-monai_q2d-retinamnist_224",
#             "242-tuning_only-example_task_retina",
#         ]
#     },
# ]

if __name__ == "__main__":
    for exp_for_dataset in automl_config[:]:
        dataset = exp_for_dataset['dataset']
        dataset_gt_path = exp_for_dataset['dataset_gt_path']
        experiment_list = exp_for_dataset['experiment_list']
        print(f"[{dataset}]")
        compute_all_metrics(dataset_gt_path, experiment_list)
        print(f"===========================================")