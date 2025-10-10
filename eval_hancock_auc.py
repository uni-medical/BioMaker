#!/usr/bin/env python3
"""
Simple AUC calculation for Hancock dataset
Usage: python calculate_hancock_auc.py
"""

import json
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Load ground truth
def load_ground_truth():
    with open('meddle/med_dl_tasks/hancock_multimodal/clinical_data.json', 'r') as f:
        data = json.load(f)
    
    # Create dataframe with binary survival (1=deceased, 0=living)
    gt = pd.DataFrame([{
        'patient_id': p['patient_id'],
        'survival': 0 if p['survival_status'] == 'deceased' else 1
    } for p in data])
    return gt

# Load predictions
def load_predictions(fname='logs/361-all_pipeline-hancock_wsi_uni/submission.csv'):
    pred = pd.read_csv(fname)
    pred['patient_id'] = pred['patient_id'].astype(str).str.zfill(3)
    return pred

# Calculate AUC
def calculate_auc(pred_fname):
    gt = load_ground_truth()
    if os.path.exists(pred_fname) is False:
        print(f"Prediction file {pred_fname} does not exist.")
        return None, None, None
    pred = load_predictions(pred_fname)
    
    # Merge on patient_id
    merged = pd.merge(gt, pred, on='patient_id')
    
    if len(merged) == 0:
        print("No matching patient IDs found")
        return
    
    # Calculate AUC
    auc = roc_auc_score(merged['survival'], merged['survival_probability'])
    
    # Convert probabilities to binary predictions (threshold 0.5)
    predictions = (merged['survival_probability'] >= 0.5).astype(int)
    
    # Calculate Accuracy and F1
    accuracy = accuracy_score(merged['survival'], predictions)
    f1 = f1_score(merged['survival'], predictions)
    
    # print(f"{auc:.4f}")
    print(f"Evaluating {fname}")
    print(f"AUC: {auc:.4f}")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"Samples: {len(merged)}")
    
    return auc, accuracy, f1

if __name__ == "__main__":
    exp_name = "2-hancock-WSI_CLAM_baseline"
    # exp_name = "2-hancock-TMA_CLAM_baseline"
    exp_name = "2-hancock-MultiModal_CLAM_baseline"
    exp_name = "2-hancock-MultiModal_CLAM_baseline_v2"
    # exp_name = "2-hancock-WSI_MedDLE_solution"
    # exp_name = "2-hancock-TMA_MedDLE_solution"
    # exp_name = "2-hancock-MultiModal_MedDLE_solution"
    # exp_name = "2-hancock-MultiModal_MedDLE_solution418"
    # exp_name = "2-hancock-MultiModal_MedDLE_solution429"
    # exp_name = "2-hancock-MultiModal_MedDLE_solution435"
    exp_name = "2-hancock-MultiModal_MedDLE_solution435_modified"
    for fname in [
        f'logs/{exp_name}/run1/submission.csv',
        # f'logs/{exp_name}/run2/submission.csv',
        # f'logs/{exp_name}/run3/submission.csv',
        # f'logs/{exp_name}/run4/submission.csv',
        # f'logs/{exp_name}/run5/submission.csv',
        # f'logs/{exp_name}/run6/submission.csv',
        # f'logs/{exp_name}/run7/submission.csv',
        # f'logs/{exp_name}/run8/submission.csv',
        # f'logs/{exp_name}/run9/submission.csv',
        # f'logs/{exp_name}/run10/submission.csv',
        # f'logs/{exp_name}/run11/submission.csv',
        # f'logs/{exp_name}/run12/submission.csv',
        # f'logs/{exp_name}/run13/submission.csv',
        # f'logs/{exp_name}/run14/submission.csv',
        # f'logs/{exp_name}/run15/submission.csv',
        # f'logs/{exp_name}/run16/submission.csv',
        # f'logs/{exp_name}/run17/submission.csv',
        # f'logs/{exp_name}/run18/submission.csv',
        # f'logs/{exp_name}/run19/submission.csv',
        # f'logs/{exp_name}/run20/submission.csv',
        'logs/2-hancock-MultiModal_CLAM_baseline_v2/run1/submission.csv',
        # baseline
        # 'logs/361-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/362-all_pipeline-hancock_wsi_uni/submission.csv',
        # new version
        # 'logs/373-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/374-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/385-all_pipeline-debug-hancock_wsi_uni/submission.csv',
        # 'logs/386-all_pipeline-debug-hancock_wsi_uni/submission.csv',
        # add tuning + hpo
        # 'logs/388-all_pipeline-debug-hancock_wsi_uni/submission.csv',
        # add tuning + hpo
        # 'logs/389-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/391-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/392-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/393-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/394-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/396-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/408-all_pipeline_TMA_only-hancock_multimodal/submission.csv', 
        # 'logs/412-all_pipeline_TMA_only-hancock_multimodal/submission.csv', 
        # 'logs/414-all_pipeline_TMA_only-hancock_multimodal/submission.csv',

        # 'logs/395-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/411-all_pipeline_WSI_only-hancock_multimodal/submission.csv', 
        # 'logs/413-all_pipeline_WSI_only-hancock_multimodal/submission.csv', 
        # 'logs/407-all_pipeline_WSI_only-hancock_multimodal/submission.csv',
        
        # 'logs/409-all_pipeline_WSI_only-hancock_multimodal/submission.csv', 
        # 'logs/410-all_pipeline_TMA_only-hancock_multimodal/submission.csv', 
        # "logs/406-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/415-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/416-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/417-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/418-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/419-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/420-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/421-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/422-all_pipeline-hancock_multimodal/submission.csv",
        # "logs/423-all_pipeline-hancock_multimodal/submission.csv",

        "logs/427-CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/428-CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/429-CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/430-CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/431-CLAM_MedDLE-hancock_multimodal/submission.csv",

        "logs/432-start_from_CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/433-start_from_CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/434-start_from_CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/435-start_from_CLAM_MedDLE-hancock_multimodal/submission.csv",
        "logs/436-start_from_CLAM_MedDLE-hancock_multimodal/submission.csv",
    ]:
        calculate_auc(fname)