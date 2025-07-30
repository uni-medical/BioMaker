import os
import warnings

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

# Suppress warnings that might arise from lifelines if data is degenerate
warnings.filterwarnings("ignore", category=UserWarning)

def compute_cindex(submission_file, ground_truth_file, survival_type='overall'):
    """
    Compute Concordance Index (C-index) for survival prediction.
    
    Parameters:
    submission_file (str): Path to submission CSV file with columns ['WSI_ID', 'Hazard_Score']
    ground_truth_file (str): Path to ground truth CSV file
    survival_type (str): Type of survival analysis - 'overall' or 'cancer_specific'
    
    Returns:
    float: C-index score
    """
    try:
        # Load submission and ground truth data
        submission_df = pd.read_csv(submission_file)
        ground_truth_df = pd.read_csv(ground_truth_file)
        
        # Determine primary key
        primary_key_candidates = ['WSI_ID', 'Id', 'ID']
        primary_key = None
        for key in primary_key_candidates:
            if key in submission_df.columns and key in ground_truth_df.columns:
                primary_key = key
                break
        if primary_key is None:
            raise ValueError(f"No valid primary key ({', '.join(primary_key_candidates)}) found in both files")
        
        # Set columns based on survival type
        if survival_type == 'cancer_specific':
            time_col = 'OS_Months'
            event_col = 'Cancer_Death_Event'
        else:  # overall
            time_col = 'OS_Months'
            event_col = 'Death_Event'
        
        # Validate required columns
        if 'Hazard_Score' not in submission_df.columns:
            raise ValueError("Submission file must contain 'Hazard_Score' column")
        if time_col not in ground_truth_df.columns:
            raise ValueError(f"Ground truth file must contain '{time_col}' column")
        if event_col not in ground_truth_df.columns:
            raise ValueError(f"Ground truth file must contain '{event_col}' column (1 for event, 0 for censored)")
        
        # Merge dataframes
        merged_df = pd.merge(
            submission_df[[primary_key, 'Hazard_Score']],
            ground_truth_df[[primary_key, time_col, event_col]],
            on=primary_key,
            how='inner'
        )
        
        if merged_df.empty:
            raise ValueError("No matching IDs found between submission and ground truth")
        
        # Extract data for C-index calculation
        event_times = merged_df[time_col].values.astype(float)
        predictions = merged_df['Hazard_Score'].values.astype(float)
        event_observed = merged_df[event_col].values.astype(int)
        
        # Calculate C-index
        c_index = concordance_index(event_times, predictions, event_observed)
        
        return c_index
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error computing C-index: {str(e)}")

def compute_all_cindex(dataset_gt_path, experiment_list, survival_type='overall'):
    """
    Compute C-index for all experiments and print results.
    
    Parameters:
    dataset_gt_path (str): Path to the ground truth CSV file
    experiment_list (list): List of experiment directory names
    survival_type (str): Type of survival analysis - 'overall' or 'cancer_specific'
    """
    results = []
    
    for exp in experiment_list:
        exp_pred_path = f"logs/{exp}/submission.csv"
        print(f"\nEvaluating: {exp_pred_path}")
        
        try:
            if not os.path.exists(exp_pred_path):
                raise FileNotFoundError(f"Submission file not found: {exp_pred_path}")
            
            c_index = compute_cindex(exp_pred_path, dataset_gt_path, survival_type)
            print(f"[{exp}] C-Index ({survival_type}): {c_index:.4f}")
            results.append({
                'Experiment': exp,
                'C-Index': c_index,
                'Survival_Type': survival_type,
                'Error': None
            })
            
        except Exception as e:
            print(f"[{exp}] Error: {str(e)}")
            results.append({
                'Experiment': exp,
                'C-Index': None,
                'Survival_Type': survival_type,
                'Error': str(e)
            })
    
    return results

# Import config from external file
from scripts.config_to_be_eval import *

### example config:
automl_cindex_config = [
    {
        'dataset': 'mco_os_prediction',
        'dataset_gt_path': 'meddle/med_dl_tasks/mco_wsi_conch/test_meta.csv',
        'survival_type': 'overall',
        'experiment_list': [
            # "1-debug-gemini_baseline-mco_os_conch",
            # "1-debug-gemini_baseline-mco_os_prediction",
            # "1-debug-gemini_ablation_fc-mco_os_conch",
            "1-debug-gemini_ablation_attenMIL-mco_os_conch",
            "350-all_pipeline-mco_os_conch",
            "355-all_pipeline-mco_os_conch",
        ]
    },
    {
        'dataset': 'mco_css_prediction',
        'dataset_gt_path': 'meddle/med_dl_tasks/mco_wsi_conch/test_meta.csv',
        'survival_type': 'cancer_specific',
        'experiment_list': [
            "1-css_attenMIL_baseline",
            "352-all_pipeline-mco_css_conch",
            "354-all_pipeline-mco_css_conch",
            "356-all_pipeline-mco_css_conch",
            "357-all_pipeline-mco_css_conch",
        ]
    },
]

if __name__ == "__main__":
    print("--- Starting C-Index Evaluation Script ---")
    
    # Use automl_cindex_config from config file
    for config in automl_cindex_config:
        dataset = config['dataset']
        dataset_gt_path = config['dataset_gt_path']
        experiment_list = config['experiment_list']
        survival_type = config.get('survival_type', 'overall')
        
        print(f"\n[{dataset}] - {survival_type.upper()} SURVIVAL")
        compute_all_cindex(dataset_gt_path, experiment_list, survival_type)
        print(f"===========================================")
    
    print("--- C-Index Evaluation Script Finished ---")