import os
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
import warnings

# Suppress warnings that might arise from lifelines if data is degenerate
warnings.filterwarnings("ignore", category=UserWarning)

# --- Existing functions (modified or kept) ---

def compute_submission_metrics(submission_file, ground_truth_file, primary_key_candidates=['Id', 'ID']):
    """
    Compute Accuracy and AUC for a submission CSV file using flexible primary key.
    This function is for standard classification tasks, NOT survival.

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
            raise ValueError(f"No valid primary key ({', '.join(primary_key_candidates)}) found in both files")

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
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_pred)

        # Computing AUC (multi-class, one-vs-rest)
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
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
    This function is for multi-label classification, NOT survival.
    
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
            raise ValueError(f"No valid primary key ({', '.join(primary_key_candidates)}) found in both files")
        
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
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true, y_pred)
            per_class_accuracy[class_col] = round(accuracy, 4)
            accuracy_scores.append(accuracy)
            
            # Compute AUC for the class
            # Ensure there are at least two classes (0 and 1) for AUC
            if len(np.unique(y_true)) > 1:
                from sklearn.metrics import roc_auc_score
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
            'accuracy': macro_accuracy, # Alias for consistency
            'per_class_auc': per_class_auc,
            'macro_auc': macro_auc,
            'auc': macro_auc, # Alias for consistency
        }
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error computing metrics: {str(e)}")


# --- New function for C-index ---
def compute_cindex_metrics(submission_file, ground_truth_file, primary_key_candidates=['WSI_ID']):
    """
    Compute Concordance Index (C-index) for a survival prediction submission CSV file.

    Parameters:
    submission_file (str): Path to submission CSV file with columns ['WSI_ID', 'Hazard_Score']
    ground_truth_file (str): Path to ground truth CSV file with columns ['WSI_ID', 'OS_Months', 'Death_Event']
    primary_key_candidates (list): List of possible primary key column names (default to WSI_ID for consistency)

    Returns:
    dict: Dictionary containing 'c_index' score
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
            raise ValueError(f"No valid primary key ({', '.join(primary_key_candidates)}) found in both files")

        # Validating required columns for survival
        if 'Hazard_Score' not in submission_df.columns:
            raise ValueError("Submission file must contain 'Hazard_Score' column")
        if 'OS_Months' not in ground_truth_df.columns:
            raise ValueError("Ground truth file must contain 'OS_Months' column")
        if 'Death_Event' not in ground_truth_df.columns:
            raise ValueError("Ground truth file must contain 'Death_Event' column (1 for event, 0 for censored)")

        # Merging dataframes on primary_key to align predictions and ground truth
        merged_df = pd.merge(
            submission_df[[primary_key, 'Hazard_Score']],
            ground_truth_df[[primary_key, 'OS_Months', 'Death_Event']],
            on=primary_key,
            how='inner' # Use inner join to only include common IDs
        )

        # Checking for missing matches
        if merged_df.empty:
            raise ValueError("No matching IDs found between submission and ground truth for C-index calculation.")

        # Extract data for C-index calculation
        # The lifelines concordance_index expects higher scores for higher risk (shorter survival)
        # and event_observed as 1 for event, 0 for censored.
        event_times = merged_df['OS_Months'].values
        predictions = merged_df['Hazard_Score'].values
        event_observed = merged_df['Death_Event'].values

        # Ensure all data types are correct for concordance_index
        event_times = event_times.astype(float)
        predictions = predictions.astype(float)
        event_observed = event_observed.astype(int)
        
        # Calculate C-index
        # lifelines.utils.concordance_index(event_times, predicted_scores, event_observed)
        c_index = concordance_index(event_times, predictions, event_observed)

        return {
            'c_index': round(c_index, 4)
        }

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error computing C-index metrics: {str(e)}")

# --- Modified compute_all_metrics to use C-index ---
def compute_all_metrics(dataset_gt_path, experiment_list):
    """
    Compute metrics for all experiments and print results.
    Detects if an experiment is for survival or classification/multilabel.

    Parameters:
    dataset_gt_path (str): Path to the ground truth CSV file
    experiment_list (list): List of experiment directory names
    """
    print(f"Ground Truth Path: {dataset_gt_path}")
    
    results = []
    
    # Pre-read ground truth to determine its type (survival, classification, multilabel)
    try:
        gt_df_peek = pd.read_csv(dataset_gt_path)
        is_survival_gt = 'OS_Months' in gt_df_peek.columns and 'Death_Event' in gt_df_peek.columns
        is_multilabel_gt = any(col.startswith('class_') for col in gt_df_peek.columns) and not is_survival_gt
        is_single_label_gt = 'class' in gt_df_peek.columns and not is_survival_gt and not is_multilabel_gt
        
        if not (is_survival_gt or is_multilabel_gt or is_single_label_gt):
            print(f"Warning: Ground truth file '{dataset_gt_path}' does not clearly indicate task type (survival, single-label, or multi-label).")
            print("Assuming single-label classification by default if 'class' column is present.")
            if 'class' in gt_df_peek.columns:
                is_single_label_gt = True

    except Exception as e:
        print(f"Error peeking into ground truth file '{dataset_gt_path}': {e}")
        return # Cannot proceed without understanding GT type

    for exp in experiment_list:
        exp_pred_path = f"logs/{exp}/submission.csv"
        print(f"\nEvaluating: {exp_pred_path}")
        
        try:
            # Check if submission file exists
            if not os.path.exists(exp_pred_path):
                raise FileNotFoundError(f"Submission file not found: {exp_pred_path}")

            # Try to infer task type from submission file if not clear from GT
            submission_df_peek = pd.read_csv(exp_pred_path)
            
            current_task_is_survival = is_survival_gt or ('Hazard_Score' in submission_df_peek.columns and 'WSI_ID' in submission_df_peek.columns)
            current_task_is_multilabel = is_multilabel_gt or any(col.startswith('class_') for col in submission_df_peek.columns) and not current_task_is_survival
            current_task_is_single_label = is_single_label_gt or ('class' in submission_df_peek.columns and not current_task_is_survival and not current_task_is_multilabel)

            metrics = {}
            if current_task_is_survival:
                print(f"[{exp}] -> Survival Task (C-index)")
                metrics = compute_cindex_metrics(exp_pred_path, dataset_gt_path)
                print(f"[{exp}] C-Index: {metrics['c_index']:.4f}")
                results.append({
                    'Experiment': exp,
                    'C-Index': metrics['c_index'],
                    'Accuracy': None, # Explicitly None for survival
                    'AUC': None,     # Explicitly None for survival
                    'Error': None
                })
            elif current_task_is_multilabel:
                print(f"[{exp}] -> Multi-label Classification Task")
                metrics = compute_multilabel_metrics(exp_pred_path, dataset_gt_path)
                print(f"[{exp}] Macro Accuracy: {metrics['macro_accuracy']:.4f}")
                print(f"[{exp}] Macro AUC: {metrics['macro_auc']:.4f}")
                results.append({
                    'Experiment': exp,
                    'C-Index': None, # Explicitly None for classification
                    'Accuracy': metrics['macro_accuracy'],
                    'AUC': metrics['macro_auc'],
                    'Error': None
                })
            elif current_task_is_single_label:
                print(f"[{exp}] -> Single-label Classification Task")
                metrics = compute_submission_metrics(exp_pred_path, dataset_gt_path)
                print(f"[{exp}] Accuracy: {metrics['accuracy']:.4f}")
                print(f"[{exp}] AUC: {metrics['auc']:.4f}")
                results.append({
                    'Experiment': exp,
                    'C-Index': None, # Explicitly None for classification
                    'Accuracy': metrics['accuracy'],
                    'AUC': metrics['auc'],
                    'Error': None
                })
            else:
                raise ValueError(f"Could not determine task type for experiment '{exp}'. Check submission/ground truth files.")

        except FileNotFoundError as e:
            print(f"[{exp}] Error: {str(e)}")
            results.append({
                'Experiment': exp,
                'C-Index': None,
                'Accuracy': None,
                'AUC': None,
                'Error': str(e)
            })
        except Exception as e:
            print(f"[{exp}] Error: {str(e)}")
            results.append({
                'Experiment': exp,
                'C-Index': None,
                'Accuracy': None,
                'AUC': None,
                'Error': str(e)
            })
    
    # Optionally, save results to a summary CSV if needed
    # pd.DataFrame(results).to_csv("evaluation_summary.csv", index=False)
    # print("\nEvaluation summary saved to evaluation_summary.csv")


# --- Configuration for evaluation ---
# Assuming scripts/config_to_be_eval.py exists and defines automl_config
# from scripts.config_to_be_eval import automl_config

# Example config for demonstration:
# Create a dummy config_to_be_eval.py or define automl_config directly here for testing
# For actual use, keep it in a separate config file as in your original script.
automl_config = [
    {
        'dataset': 'mco_os_prediction',
        'dataset_gt_path': 'meddle/med_dl_tasks/mco_wsi_gigapath/test_meta.csv',
        'experiment_list': [
            "1-debug-gemini_baseline-mco_os_conch",
            "1-debug-gemini_baseline-mco_os_prediction",
            "1-debug-gemini_ablation_fc-mco_os_conch",
            "1-debug-gemini_ablation_attenMIL-mco_os_conch",
            "344-all_pipeline-mco_os_conch",
            # "338-debug-mco_os_prediction",
            # "339-all_pipeline-mco_os_prediction",
            # "339-all_pipeline-mco_os_prediction_negative_loss",
            # "342-all_pipeline-mco_os_conch",
        ]
    },
]


if __name__ == "__main__":
    print("--- Starting Evaluation Script ---")
    for exp_for_dataset in automl_config[:]:
        dataset = exp_for_dataset['dataset']
        dataset_gt_path = exp_for_dataset['dataset_gt_path']
        experiment_list = exp_for_dataset['experiment_list']
        print(f"\n[{dataset}]")
        compute_all_metrics(dataset_gt_path, experiment_list)
        print(f"===========================================")
    print("--- Evaluation Script Finished ---")