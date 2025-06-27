meddle data_dir="meddle/example_tasks/retinamnist_224" exp_name="torch-example_task_retina"\
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5  agent.enable_monai_knowledge_base=true agent.expose_prediction=true 
