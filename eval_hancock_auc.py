#!/usr/bin/env python3
"""
Simple AUC calculation for Hancock dataset
Usage: python calculate_hancock_auc.py
"""

import json
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Load ground truth
def load_ground_truth():
    with open('meddle/med_dl_tasks/hancock_wsi_uni/clinical_data.json', 'r') as f:
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
    
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Samples: {len(merged)}")
    
    return auc, accuracy, f1

if __name__ == "__main__":
    for fname in [
        'logs/361-all_pipeline-hancock_wsi_uni/submission.csv',
        'logs/362-all_pipeline-hancock_wsi_uni/submission.csv',
        # new version
        'logs/373-all_pipeline-hancock_wsi_uni/submission.csv',
        'logs/374-all_pipeline-hancock_wsi_uni/submission.csv',
        # add tuning + hpo
        'logs/375-all_pipeline-hancock_wsi_uni/submission.csv',
        'logs/376-all_pipeline-hancock_wsi_uni/submission.csv',
        'logs/377-all_pipeline-hancock_wsi_uni/submission.csv',
    ]:
        print(f"Evaluating {fname}")
        calculate_auc(fname)