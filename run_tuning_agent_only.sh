meddle data_dir="meddle/example_tasks/retinamnist_224" exp_name="tuning_only-example_task_retina" \
     proposed_solution_file="exp_summary/best_solutions/92-monai_q2d-retinamnist_224.py" \
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=10 agent.steps_for_tuning=10 agent.enable_monai_knowledge_base=true

meddle data_dir="meddle/example_tasks/retinamnist_224" exp_name="tuning_only-example_task_retina" \
     proposed_solution_file="exp_summary/best_solutions/92-monai_q2d-retinamnist_224.py" \
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.steps_for_tuning=20 agent.enable_monai_knowledge_base=true

meddle data_dir="meddle/example_tasks/retinamnist_224" exp_name="tuning_only-example_task_retina" \
     proposed_solution_file="exp_summary/best_solutions/92-monai_q2d-retinamnist_224.py" \
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=30 agent.steps_for_tuning=30 agent.enable_monai_knowledge_base=true