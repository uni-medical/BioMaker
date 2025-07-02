# meddle data_dir="meddle/med_dl_tasks/organmnist3d_64" exp_name="all_pipeline-organmnist3d_64"\
#      goal="Predict the category of each given 3D Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
#      eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
#      agent.steps=30 agent.enable_monai_knowledge_base=true agent.expose_prediction=true 
     
# meddle data_dir="meddle/med_dl_tasks/organmnist3d_64" exp_name="all_pipeline-organmnist3d_64"\
#      goal="Predict the category of each given 3D Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
#      eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
#      agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.enable_monai_knowledge_base=true agent.enable_query2doc=true

meddle data_dir="meddle/med_dl_tasks/nodulemnist3d_64" exp_name="all_pipeline-nodulemnist3d_64"\
     goal="Predict the category of each given 3D Chest CT image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 model_for_all_agents="gpt-4o-mini" log_level="DEBUG"
     
# meddle data_dir="meddle/med_dl_tasks/breastmnist_224" exp_name="all_pipeline-breastmnist_224"\
#      goal="Predict the category of each given Breast Ultrasound image by selecting the most appropriate one from the 2 available classes." \
#      eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
#      agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.enable_monai_knowledge_base=true agent.enable_query2doc=true

# meddle data_dir="meddle/med_dl_tasks/retinamnist_224" exp_name="all_pipeline-example_task_retina"\
#      goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
#      eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
#      agent.step_plan.proposal=10 agent.step_plan.tuning=5 agent.step_plan.hpo=2 agent.enable_monai_knowledge_base=true agent.expose_prediction=true 

# python -m meddle.run \
#     data_dir="meddle/med_dl_tasks/breastmnist_224" exp_name="tuning_only-breastmnist_224" \
#     proposed_solution_file="exp_summary/best_solutions/93-monai_q2d-breastmnist_224.py" \
#     goal="Predict the category of each given Breast Ultrasound image by selecting the most appropriate one from the 2 available classes." \
#     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
#     agent.step_plan.proposal=0 agent.step_plan.tuning=3 agent.step_plan.hpo=1 agent.enable_monai_knowledge_base=true generate_report=false

# python meddle/run.py \
#     data_dir="meddle/med_dl_tasks/retinamnist_224" exp_name="tuning_only-example_task_retina" \
#     proposed_solution_file="exp_summary/best_solutions/92-monai_q2d-retinamnist_224.py" \
#     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
#     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
#     agent.step_plan.proposal=0 agent.step_plan.tuning=3 agent.step_plan.hpo=1 agent.enable_monai_knowledge_base=true   