#!/usr/bin/env python3
"""
Simple AUC calculation and ROC data extraction for Hancock dataset.
Exports data to a long-format Excel file with three columns: FPR, Label, TPR.
For each label and each unique FPR, only the maximum TPR is kept.
"""

import json
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, roc_curve

# --- Helper functions (no changes) ---

def load_ground_truth():
    """Loads and prepares the ground truth data."""
    with open('meddle/med_dl_tasks/hancock_multimodal/clinical_data.json', 'r') as f:
        data = json.load(f)
    gt = pd.DataFrame([{
        'patient_id': p['patient_id'],
        'survival': 1 if p['survival_status'] == 'living' else 0
    } for p in data])
    return gt

def load_predictions(fname):
    """Loads and prepares a prediction file."""
    pred = pd.read_csv(fname)
    pred['patient_id'] = pred['patient_id'].astype(str).str.zfill(3)
    return pred

# --- Analysis function (no changes) ---

def analyze_predictions(pred_fname):
    """Calculates metrics and raw ROC curve points for a single prediction file."""
    gt = load_ground_truth()
    if not os.path.exists(pred_fname):
        print(f"  -> Prediction file not found.")
        return None
    
    pred = load_predictions(pred_fname)
    merged = pd.merge(gt, pred, on='patient_id')
    
    if len(merged) == 0:
        print("  -> No matching patient IDs found.")
        return None
    
    y_true = merged['survival']
    y_score = merged['survival_probability']
    
    auc = roc_auc_score(y_true, y_score)
    print(f"  -> AUC: {auc:.4f}, Samples: {len(merged)}")
    
    # Get the raw ROC curve points
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    return {"fpr": fpr, "tpr": tpr}

# --- Main execution block (MODIFIED) ---

if __name__ == "__main__":
    prediction_files = [
        'logs/2-hancock-wsi_clam_baseline/submission.csv',
        'logs/2-hancock-TMA-CLAM_baseline/submission.csv',
        # 'logs/392-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/393-all_pipeline-hancock_wsi_uni/submission.csv',
        # 'logs/394-all_pipeline-hancock_wsi_uni/submission.csv',
        ('logs/395-all_pipeline-hancock_wsi_uni/submission.csv', "WSI"),
        ('logs/410-all_pipeline_TMA_only-hancock_multimodal/submission.csv', 'TMA'),
        ("logs/406-all_pipeline-hancock_multimodal/submission.csv", "WSI+TMA"),
        # 'logs/407-all_pipeline_WSI_only-hancock_multimodal/submission.csv', 
        # 'logs/408-all_pipeline_TMA_only-hancock_multimodal/submission.csv', 
        # 'logs/409-all_pipeline_WSI_only-hancock_multimodal/submission.csv', 
        # 'logs/410-all_pipeline_TMA_only-hancock_multimodal/submission.csv', 
        # 'logs/411-all_pipeline_WSI_only-hancock_multimodal/submission.csv', 
        # 'logs/412-all_pipeline_TMA_only-hancock_multimodal/submission.csv', 
        # 'logs/413-all_pipeline_WSI_only-hancock_multimodal/submission.csv', 
    ]

    # A list to hold the DataFrame for each experiment
    list_of_dfs = []

    for fname in prediction_files:
        # Use the folder name as the experiment label
        if isinstance(fname, str):
            experiment_label = os.path.basename(os.path.dirname(fname))
        else:
            experiment_label = fname[1]
            fname = fname[0]

        print(f"Processing: {fname}")
        results = analyze_predictions(fname)
        
        if results:
            # Create a temporary DataFrame from the raw ROC points
            raw_df = pd.DataFrame({
                'FPR': results['fpr'],
                'TPR': results['tpr']
            })

            # For each unique FPR, find the index of the maximum TPR.
            # This ensures only one, best TPR is kept per FPR level.
            filtered_df = raw_df.loc[raw_df.groupby('FPR')['TPR'].idxmax()]
            
            # Add the label column and ensure correct column order
            final_temp_df = filtered_df.copy() # Use .copy() to avoid warnings
            final_temp_df['Label'] = experiment_label
            final_temp_df = final_temp_df[['FPR', 'Label', 'TPR']]

            list_of_dfs.append(final_temp_df)

    # Check if any data was collected
    if list_of_dfs:
        # Concatenate all DataFrames vertically into a single one
        final_df = pd.concat(list_of_dfs, ignore_index=True)
        
        output_filename = 'roc_results_long_format_unique.xlsx'
        
        try:
            final_df.to_excel(output_filename, index=False, engine='openpyxl')
            print(f"\n✅ All ROC curve data successfully saved to: {output_filename}")
            print("   Format: Filtered to keep only the maximum TPR for each FPR per experiment.")
        except ImportError:
            print("\n⚠️ Could not save to Excel. The 'openpyxl' library is required.")
            print("   Please install it by running: pip install openpyxl pandas")
    else:
        print("\nNo data was processed to save.")