meddle data_dir="/medmnist_pre/pathmnist_224" exp_name="monai_q2d-pathmnist_224"\
     goal="Predict the category of each given Colon Pathology image by selecting the most appropriate one from the 9 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true 

meddle data_dir="/medmnist_pre/chestmnist_224" exp_name="monai_q2d-chestmnist_224" \
     goal="This is a multi-label classification task. Predict the category of each given 2D Chest X-Ray image by selecting the categories present in the image. There are a total of 14 possible categories, and each image can contain more than one category. Use 1 to indicate the presence of a category and 0 to indicate its absence." \
     eval="Use the accuracy and the area under curve (AUC) appied in a multi-label classification task on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true 

meddle data_dir="/medmnist_pre/retinamnist_224" exp_name="monai_q2d-retinamnist_224" \
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true 


meddle data_dir="/medmnist_pre/breastmnist_224" exp_name="monai_q2d-breastmnist_224" \
     goal="Predict the category of each given Breast Ultrasound image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true 


meddle data_dir="/medmnist_pre/bloodmnist_224" exp_name="monai_q2d-bloodmnist_224" \
     goal="Predict the category of each given Blood Cell Microscope image by selecting the most appropriate one from the 8 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true 


meddle data_dir="/medmnist_pre/tissuemnist_224" exp_name="monai_q2d-tissuemnist_224" \
     goal="Predict the category of each given Kidney Cortex Microscope image by selecting the most appropriate one from the 8 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true 


meddle data_dir="/medmnist_pre/organamnist_224" exp_name="monai_q2d-organamnist_224" \
     goal="Predict the category of each given Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true  agent.expose_prediction=true 


meddle data_dir="/medmnist_pre/organcmnist_224" exp_name="monai_q2d-organcmnist_224" \
     goal="Predict the category of each given Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true  agent.expose_prediction=true 


meddle data_dir="/medmnist_pre/organsmnist_224" exp_name="monai_q2d-organsmnist_224" \
     goal="Predict the category of each given Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true  agent.expose_prediction=true 