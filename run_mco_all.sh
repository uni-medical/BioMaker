# meddle data_dir="meddle/med_dl_tasks/mco_wsi_gigapath" exp_name="debug-mco_os_prediction" \
# meddle data_dir="meddle/med_dl_tasks/mco_wsi_gigapath" exp_name="all_pipeline-mco_os_gigapath" \
#      goal="Based on a patient's whole slide images (WSIs), predict their survival risk (hazard score) to stratify them into high-risk and low-risk prognostic groups." \
#      eval="Evaluate model performance using the Concordance Index (C-Index) to measure risk prediction accuracy. Additionally, use Kaplan-Meier survival curves and the Log-rank test to assess if the stratification between predicted high-risk and low-risk groups is statistically significant (P-Value < 0.05)." \

# meddle data_dir="meddle/med_dl_tasks/mco_wsi_gigapath" exp_name="all_pipeline-mco_os_gigapath" \
meddle data_dir="meddle/med_dl_tasks/mco_wsi_conch" exp_name="all_pipeline-mco_os_conch" \
     goal="Predict patient survival risk (hazard score) from whole slide image (WSI) features (shape = tile_num x feat_dim)." \
     eval="Use the Concordance Index (C-Index) to evaluate risk prediction accuracy." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=0 agent.force_monai_with_prompt=false


