meddle data_dir="meddle/example_tasks/retinamnist_224" exp_name="torch-example_task_retina"\
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 \
     model_for_all_agents="deepseek-r1"
     # model_for_all_agents="deepseek-v3"
     # model_for_all_agents="gpt-4o-mini"
     # model_for_all_agents="claude-3-7-sonnet-20250219"
     # model_for_all_agents="o1-preview-2024-09-12"
     # model_for_all_agents="claude-3-5-haiku-20241022"