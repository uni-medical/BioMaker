# Med-DLE: Agentic Medical Deep Learning Engineer 

# Quick Start
## Basic
```
pip install -r requirements.txt
```
You need to get data from [link](https://ug.link/haoyushare/filemgr/share-download/?id=7a3a2c6beacb4d5e91a2d801bc653bce) and move the data into `med_dl_tasks`

Then you can use `bash run.sh` to run the demo experiment.

## Advanced
We support MONAI with prompt/RAG
### prompt
```
meddle data_dir="meddle/med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.steps=20 agent.force_monai_with_prompt=true 
```
### RAG
### Create the knowledge database
Firstly, download the persisted database into `meddle/monai_rag` from [link](http://ug.link/haoyushare/filemgr/share-download/?id=0114fe77cb51407ba873896a004d1b9e). 
Or you can create the db with this cmd:
```
cd meddle/monai_rag
python create_monai_rag_db.py
```
You could use `python -m meddle.monai_rag.query_rag_db` to check whether the creation succeeds.

### enable the feature in agent
Here's some example:
```
meddle data_dir="med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt_kb-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true

meddle data_dir="med_dl_tasks/odir5k_2d_mlc" exp_name="monai_prompt_kb_q2d-odir5k_2d_mlc"\
     goal="Classify 2D Medical images into different categories in a multi-label setting" \
     eval="Use the average of kappa score, F-1 score, and AUC value metric between the predicted and ground-truth values." \
     agent.steps=20 agent.force_monai_with_prompt=false agent.enable_monai_knowledge_base=true agent.enable_query2doc=true
```

