python -m meddle.run \
    data_dir="meddle/med_dl_tasks/breastmnist_224" exp_name="tuning_only-breastmnist_224" \
    proposed_solution_file="exp_summary/best_solutions/93-monai_q2d-breastmnist_224.py" \
    goal="Predict the category of each given Breast Ultrasound image by selecting the most appropriate one from the 2 available classes." \
    eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
    agent.step_plan.proposal=0 agent.step_plan.tuning=3 agent.step_plan.hpo=1 agent.enable_monai_knowledge_base=true generate_report=false

# python meddle/run.py \
#     data_dir="meddle/med_dl_tasks/retinamnist_224" exp_name="tuning_only-example_task_retina" \
#     proposed_solution_file="exp_summary/best_solutions/92-monai_q2d-retinamnist_224.py" \
#     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
#     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
#     agent.step_plan.proposal=0 agent.step_plan.tuning=3 agent.step_plan.hpo=1 agent.enable_monai_knowledge_base=true   