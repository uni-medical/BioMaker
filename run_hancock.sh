# meddle data_dir="meddle/med_dl_tasks/mco_wsi_gigapath" exp_name="debug-mco_os_prediction" \
# meddle data_dir="meddle/med_dl_tasks/mco_wsi_gigapath" exp_name="all_pipeline-mco_os_gigapath" \
#      goal="Based on a patient's whole slide images (WSIs), predict their survival risk (hazard score) to stratify them into high-risk and low-risk prognostic groups." \
#      eval="Evaluate model performance using the Concordance Index (C-Index) to measure risk prediction accuracy. Additionally, use Kaplan-Meier survival curves and the Log-rank test to assess if the stratification between predicted high-risk and low-risk groups is statistically significant (P-Value < 0.05)." \

# HANCOCK paper 里汇报的WSI结果是random 80:20 split的结果，并且无split文件，但是复现in/out等官方split的多模态结果比较复杂
# 
meddle data_dir="meddle/med_dl_tasks/hancock_wsi_uni" exp_name="all_pipeline-hancock_wsi_uni" \
     goal="Predict patient survival status (living/deceased) from whole slide image (WSI) features. The feature is saved under the key 'features' of h5 files (shape = tile_num x feat_dim)." \
     eval="Use AUC on self-hold split of training set to evaluate accuracy." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
     agent.use_deep_research_knowledge="wsi_tricks"

# meddle data_dir="meddle/med_dl_tasks/mco_wsi_conch" exp_name="all_pipeline-mco_css_conch" \
#      goal="Predict patient Cancer-Specific Survival risk (hazard score) from whole slide image (WSI) features (shape = tile_num x feat_dim)." \
#      eval="Use the Concordance Index (C-Index) based on OS_Months and Cancer_Death_Event to evaluate risk prediction accuracy" \
#      agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
#      agent.force_monai_with_prompt=false agent.use_deep_research_knowledge="wsi_tricks"
