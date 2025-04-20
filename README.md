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
3. Get data from [link](https://ug.link/haoyushare/filemgr/share-download/?id=7a3a2c6beacb4d5e91a2d801bc653bce) and move the data into `med_dl_tasks`

4. remember to set up your private api key of OPENAI/ANTROPIC/OPENROUNTER
```
export OPENAI_BASE_URL="<your base url>" # (e.g. https://api.openai.com/v1)
export OPENAI_API_KEY="<your key>"
```

## Basic codegen

Use `bash run.sh` to run the demo experiment.

## MONAI-enhanced codegen
We highlight MONAI as a helpful candidate for medical DL codegen with Prompt/RAG mode:
### Prompt mode
```
meddle data_dir="meddle/med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.steps=20 agent.force_monai_with_prompt=true 
```
### RAG mode
#### 1. Create the knowledge database
Firstly, download the persisted database into `meddle/monai_rag` from [link](http://ug.link/haoyushare/filemgr/share-download/?id=0114fe77cb51407ba873896a004d1b9e). 
Or you can create the db with this cmd:
```
cd meddle/monai_rag
python create_monai_rag_db.py
```
You could use `python -m meddle.monai_rag.query_rag_db` to check whether the creation succeeds.

#### 2. enable the feature in agent
Here's some example:
```
meddle data_dir="meddle/med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt_kb-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true

meddle data_dir="meddle/med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt_kb_q2d-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true
```


## 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects, our code is developed based on their contributions:
     - [aideml](https://github.com/WecoAI/aideml)
     - [monai](https://github.com/Project-MONAI)