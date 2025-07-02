#!/bin/bash

# aide_monai_rag_q2d 配置的所有实验

# pathmnist_224 实验
meddle data_dir="meddle/med_dl_tasks/pathmnist_224" exp_name="monai_rag_q2d-pathmnist_224-20p10t5h" \
     goal="Predict the category of each given Colon Pathology image by selecting the most appropriate one from the 9 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# dermamnist_224 实验
meddle data_dir="meddle/med_dl_tasks/dermamnist_224" exp_name="monai_rag_q2d-dermamnist_224-20p10t5h" \
     goal="Predict the category of each given Dermatoscope image by selecting the most appropriate one from the 7 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# octmnist_224 实验
meddle data_dir="meddle/med_dl_tasks/octmnist_224" exp_name="monai_rag_q2d-octmnist_224-20p10t5h" \
     goal="Predict the category of each given Retinal OCT image by selecting the most appropriate one from the 4 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# pneumoniamnist_224 实验
meddle data_dir="meddle/med_dl_tasks/pneumoniamnist_224" exp_name="monai_rag_q2d-pneumoniamnist_224-20p10t5h" \
     goal="Predict the category of each given Chest X-Ray image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# chestmnist_224 实验
meddle data_dir="meddle/med_dl_tasks/chestmnist_224" exp_name="monai_rag_q2d-chestmnist_224-20p10t5h" \
     goal="This is a multi-label classification task. Predict the category of each given 2D Chest X-Ray image by selecting the categories present in the image. There are a total of 14 possible categories, and each image can contain more than one category. Use 1 to indicate the presence of a category and 0 to indicate its absence." \
     eval="Use the accuracy and the area under curve (AUC) appied in a multi-label classification task on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# retinamnist_224 实验
meddle data_dir="meddle/med_dl_tasks/retinamnist_224" exp_name="monai_rag_q2d-retinamnist_224-20p10t5h" \
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# breastmnist_224 实验
meddle data_dir="meddle/med_dl_tasks/breastmnist_224" exp_name="monai_rag_q2d-breastmnist_224-20p10t5h" \
     goal="Predict the category of each given Breast Ultrasound image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# bloodmnist_224 实验
meddle data_dir="meddle/med_dl_tasks/bloodmnist_224" exp_name="monai_rag_q2d-bloodmnist_224-20p10t5h" \
     goal="Predict the category of each given Blood Cell Microscope image by selecting the most appropriate one from the 8 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# tissuemnist_224 实验
meddle data_dir="meddle/med_dl_tasks/tissuemnist_224" exp_name="monai_rag_q2d-tissuemnist_224-20p10t5h" \
     goal="Predict the category of each given Kidney Cortex Microscope image by selecting the most appropriate one from the 8 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# organamnist_224 实验
meddle data_dir="meddle/med_dl_tasks/organamnist_224" exp_name="monai_rag_q2d-organamnist_224-20p10t5h" \
     goal="Predict the category of each given Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# organcmnist_224 实验
meddle data_dir="meddle/med_dl_tasks/organcmnist_224" exp_name="monai_rag_q2d-organcmnist_224-20p10t5h" \
     goal="Predict the category of each given Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# organsmnist_224 实验
meddle data_dir="meddle/med_dl_tasks/organsmnist_224" exp_name="monai_rag_q2d-organsmnist_224-20p10t5h" \
     goal="Predict the category of each given Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# organmnist3d_64 实验
meddle data_dir="meddle/med_dl_tasks/organmnist3d_64" exp_name="monai_rag_q2d-organmnist3d_64-20p10t5h" \
     goal="Predict the category of each given 3D Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# nodulemnist3d_64 实验
meddle data_dir="meddle/med_dl_tasks/nodulemnist3d_64" exp_name="monai_rag_q2d-nodulemnist3d_64-20p10t5h" \
     goal="Predict the category of each given 3D Chest CT image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# adrenalmnist3d_64 实验
meddle data_dir="meddle/med_dl_tasks/adrenalmnist3d_64" exp_name="monai_rag_q2d-adrenalmnist3d_64-20p10t5h" \
     goal="Predict the category of each given 3D Shape from Abdominal CT image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# fracturemnist3d_64 实验
meddle data_dir="meddle/med_dl_tasks/fracturemnist3d_64" exp_name="monai_rag_q2d-fracturemnist3d_64-20p10t5h" \
     goal="Predict the category of each given 3D Chest CT image by selecting the most appropriate one from the 3 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# vesselmnist3d_64 实验
meddle data_dir="meddle/med_dl_tasks/vesselmnist3d_64" exp_name="monai_rag_q2d-vesselmnist3d_64-20p10t5h" \
     goal="Predict the category of each given 3D Shape from Brain MRA image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true

# synapsemnist3d_64 实验
meddle data_dir="meddle/med_dl_tasks/synapsemnist3d_64" exp_name="monai_rag_q2d-synapsemnist3d_64-20p10t5h" \
     goal="Predict the category of each given 3D Electron Microscope image by selecting the most appropriate one from the 2 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5 agent.force_monai_with_prompt=true agent.enable_monai_knowledge_base=true agent.enable_query2doc=true agent.expose_prediction=true
