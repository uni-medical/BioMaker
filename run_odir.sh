# meddle data_dir="med_dl_tasks/odir5k_2d_mlc" exp_name="base-odir5k_2d_mlc"\
#      goal="Classify 2D Medical images into different categories in a multi-label setting" \
#      eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
#      agent.steps=20 agent.force_monai_with_prompt=false 

# meddle data_dir="med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt-odir5k_2d_mlc"\
#      goal="Classify 2D Medical images into different categories in a multi-label setting" \
#      eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
#      agent.steps=20 agent.force_monai_with_prompt=true 

# meddle data_dir="med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt_kb-odir5k_2d_mlc"\
#      goal="Classify 2D Medical images into different categories in a multi-label setting" \
#      eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
#      agent.steps=20 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true

meddle data_dir="med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt_kb_q2d-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.steps=20 agent.expose_prediction=true agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true

# meddle data_dir="med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt-odir5k_2d_mlc"\
#      goal="Classify 2D Medical images into different categories in a multi-label setting" \
#      eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
#      agent.steps=20 agent.expose_prediction=true agent.force_monai_with_prompt=true 