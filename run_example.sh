
meddle data_dir="meddle/example_tasks/retinamnist_224" \
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=10 agent.force_monai_with_prompt=true agent.expose_prediction=true 