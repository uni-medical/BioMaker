#!/bin/bash

meddle data_dir="meddle/med_dl_tasks/dermamnist_224" exp_name="meddle_all-dermamnist_224-gpt4p1" \
     goal="Predict the category of each given Dermatoscope image by selecting the most appropriate one from the 7 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     model_for_all_agents="gpt-4.1" \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5

meddle data_dir="meddle/med_dl_tasks/retinamnist_224" exp_name="meddle_all-retinamnist_224-gpt4p1" \
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     model_for_all_agents="gpt-4.1" \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5

meddle data_dir="meddle/med_dl_tasks/breastmnist_224" exp_name="meddle_all-breastmnist_224-gpt4p1" \
     goal="Predict the category of each given Breast Ultrasound image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     model_for_all_agents="gpt-4.1" \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5

meddle data_dir="meddle/med_dl_tasks/tissuemnist_224" exp_name="meddle_all-tissuemnist_224-gpt4p1" \
     goal="Predict the category of each given Kidney Cortex Microscope image by selecting the most appropriate one from the 8 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     model_for_all_agents="gpt-4.1" \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5

meddle data_dir="meddle/med_dl_tasks/nodulemnist3d_64" exp_name="meddle_all-nodulemnist3d_64-gpt4p1" \
     goal="Predict the category of each given 3D Chest CT image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     model_for_all_agents="gpt-4.1" \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5

meddle data_dir="meddle/med_dl_tasks/fracturemnist3d_64" exp_name="meddle_all-fracturemnist3d_64-gpt4p1" \
     goal="Predict the category of each given 3D Chest CT image by selecting the most appropriate one from the 3 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     model_for_all_agents="gpt-4.1" \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5