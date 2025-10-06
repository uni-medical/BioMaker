# HANCOCK paper 里汇报的WSI结果是random 80:20 split的结果，并且无split文件，但是复现in/out等官方split的多模态结果比较复杂
# 这里直接使用了in distribution的官方split作为split，执行WSI only的任务
# for i in {0..3}; do
# meddle data_dir="meddle/med_dl_tasks/hancock_wsi_uni" exp_name="all_pipeline-hancock_wsi_uni" \
#      goal="Predict patient survival status (living/deceased) from whole slide image (WSI) features." \
#      eval="Split the training data into 80% for training and 20% for validation. Use AUC on self-hold split of training set to evaluate accuracy. " \
#      agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
#      agent.use_deep_research_knowledge="wsi_tricks" log_level="DEBUG"
# done

# for i in {0..3}; do
# meddle data_dir="meddle/med_dl_tasks/hancock_multimodal" exp_name="all_pipeline_WSI_only-hancock_multimodal" \
#      goal="Predict patient survival status (living/deceased) based on features. You are only allowed to use WSI features." \
#      eval="Use AUC and Accuracy to evaluate the results. " \
#      agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
#      agent.use_deep_research_knowledge="wsi_tricks" log_level="DEBUG" model_for_all_agents="gpt-4.1"

# meddle data_dir="meddle/med_dl_tasks/hancock_multimodal" exp_name="all_pipeline_TMA_only-hancock_multimodal" \
#      goal="Predict patient survival status (living/deceased) based on features. You are only allowed to use TMA features." \
#      eval="Use AUC and Accuracy to evaluate the results. " \
#      agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
#      agent.use_deep_research_knowledge="wsi_tricks" log_level="DEBUG" model_for_all_agents="gpt-4.1"
# done

# for i in {0..3}; do
# meddle data_dir="meddle/med_dl_tasks/hancock_multimodal" exp_name="CLAM_MedDLE-hancock_multimodal" \
#      goal="Predict patient survival status (living/deceased) based on multimodal features (WSI+TMA). You are recommanded to use CLAM as the baseline." \
#      eval="Use AUC on self-hold split of training set for evaluation." \
#      agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
#      agent.use_deep_research_knowledge="wsi_tricks" log_level="DEBUG" model_for_all_agents="gpt-4.1"
# done

for i in {0..3}; do
meddle data_dir="meddle/med_dl_tasks/hancock_multimodal" exp_name="start_from_CLAM_MedDLE-hancock_multimodal" \
     goal="Predict patient survival status (living/deceased) based on multimodal features (WSI+TMA). You are recommanded to use CLAM as the baseline." \
     eval="Use AUC on self-hold split of training set for evaluation." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
     agent.use_deep_research_knowledge="wsi_tricks" log_level="DEBUG" model_for_all_agents="gpt-4.1"
     # proposed_solution_file="/home/haoyu/code/MedDLE/workspaces/0-workspace-hancock_multimodal/CLAM_multimodal_baseline_for_MedDLE.py"
done