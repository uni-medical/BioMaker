# HANCOCK paper 里汇报的WSI结果是random 80:20 split的结果，并且无split文件，但是复现in/out等官方split的多模态结果比较复杂
# 这里直接使用了in distribution的官方split作为split，执行WSI only的任务
for i in {0..3}; do
meddle data_dir="meddle/med_dl_tasks/hancock_wsi_uni" exp_name="all_pipeline-hancock_wsi_uni" \
     goal="Predict patient survival status (living/deceased) from whole slide image (WSI) features." \
     eval="Split the training data into 80% for training and 20% for validation. Use AUC on self-hold split of training set to evaluate accuracy. " \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
     agent.use_deep_research_knowledge="wsi_tricks" log_level="DEBUG"
done