# Med-DLE: Agentic Medical Deep Learning Engineer 

MedDLE is designed to be a codegen agentic system for medical deep learning tasks. 

# Quick Start
## Setup environment
1. we recommand use `conda` to create a virtual environment:
```
conda create --name meddle python=3.11 
conda activate meddle
```
2. Then install `meddle` via:
```
pip install -r requirements.txt
pip install -e .
```
> optional: use `uv` to speedup installation
> ```
> pip install uv
> uv pip install -r requirements.txt
> uv pip install -e .
> ```
3. remember to set up your private api key of OPENAI/ANTROPIC/OPENROUNTER
```
export OPENAI_BASE_URL="<your base url>" # (e.g. https://api.openai.com/v1)
export OPENAI_API_KEY="<your key>"
```
### Setup knowledge base for RAG
You can create the db with `create_monai_knowledge_base.sh` or download from [link](http://ug.link/haoyushare/filemgr/share-download/?id=b8c77eb1346c4ecca4dfb510253d32fe).

You could use this command to check whether the creation succeeds.
```
python -m meddle.monai_rag.query_rag_db
``` 

### Setup dataset
1. Get data from [link](http://ug.link/haoyushare/filemgr/share-download/?id=a14cf769efb649f8a099ee6fb876abb6) and move the data into `med_dl_tasks`
2. The goal and eval target of MedMNIST dataset can be found in [scripts](https://github.com/uni-medical/MedDLE/blob/main/scripts/run_all_mnist_rag.sh).

## Try MedDLE
```
meddle data_dir=<your dataset> \
     goal=<your goal> \
     eval=<your metrics> \
     agent.steps=30 agent.enable_monai_knowledge_base=true
```
here's a detailed example:
```
meddle data_dir="meddle/example_tasks/retinamnist_224" exp_name="torch-example_task_retina"\
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.step_plan.proposal=20 agent.step_plan.tuning=10 agent.step_plan.hpo=5  \
     agent.enable_monai_knowledge_base=true agent.expose_prediction=true 
```
You can set general `steps` (MedDLE with auto schedule multiple agents' step) or detailed `step_plan` for 
each agent.

## MONAI-enhanced codegen
We highlight MONAI as a helpful candidate for medical DL codegen with Prompt/RAG mode:
### Prompt mode
```
meddle data_dir="meddle/med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.force_monai_with_prompt=true 
```
### RAG mode
#### 1. enable the feature in agent
We support both basic RAG and advanced (query2doc) mode.
```
meddle data_dir="meddle/med_dl_tasks/odir5k_2d_mlc" exp_name="monai_rag-odir5k"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.enable_monai_knowledge_base=true

meddle data_dir="meddle/med_dl_tasks/odir5k_2d_mlc" exp_name="monai_q2d-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.enable_monai_knowledge_base=true agent.enable_query2doc=true
```

### Details
To test with a total time limit, you can set config `agent.total_time_limit`:
```
meddle data_dir="meddle/example_tasks/retinamnist_224" exp_name="test_timelimit-example_task_retina"\
     goal="Predict the category of each given Fundus Camera image by selecting the most appropriate one from the 5 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set." \
     agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=false agent.expose_prediction=true \
     agent.steps=100 agent.total_time_limit=60
```

## 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects, our code is developed based on their contributions:
     - [aideml](https://github.com/WecoAI/aideml)
     - [monai](https://github.com/Project-MONAI)