<div align="center" id="creatop">
<img src="./assets/CREA_logo.png" alt="logo" width="400" margin="10px"></img>
<br/>
<a href="https://pypi.org/project/crea-med"><img src="https://img.shields.io/pypi/v/crea-med" alt="PyPI"></a>
<a href="https://github.com/uni-medical/CREA/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="license"></a>
<img src="https://img.shields.io/badge/Python-3.11-A7D8FF" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.9-FCD299" alt="PyTorch">
<img src="https://img.shields.io/badge/Paper-to%20be%20released-b31b1b?logo=arxiv&logoColor=red" alt="Paper">
</div>

# Introduction
**CREA** is designed to be a **C**ollaborative Failure-aware **RE**asoning **A**gent for Biomedical AutoML.

# Quick Start

## Setup environment

We recommend using `uv` to manage the environment and dependencies.

1. **Install uv** (if you haven't already):
     ```bash
     pip install uv
     ```
     or using the standalone installer:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

2. **Create a virtual environment with Python 3.11**:
     ```bash
     uv venv --python 3.11
     source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
     ```

3. **Install dependencies**:
     ```bash
     git clone https://github.com/uni-medical/CREA
     cd CREA
     uv pip install -r requirements.txt
     uv pip install -e .
     ```

4. **Set up API keys**:
Set up your private API key for OPENAI-compatible endpoints (e.g. openai official, openrouter):
     ```bash
     export OPENAI_BASE_URL="<your base url>" # (e.g. https://api.openai.com/v1)
     export OPENAI_API_KEY="<your key>"
     ```
5. **Setup knowledge base for RAG Tool**
You can create the knowledge base with `bash scripts/create_rag_db.sh`.
For MONAI-1.4.0, you can also download from [Huggingface](https://huggingface.co/datasets/blueyo0/monai_rag_db). (`bash scripts/download_rag_db.sh`)

     You could use this command to check whether the creation succeeds.
     ```
     python -m crea.monai_rag.query_rag_db
     ``` 
## Test with Example Dataset
You can download the example dataset from [Huggingface](https://huggingface.co/datasets/blueyo0/organmnist3d).
Download command:
```
hf download blueyo0/organmnist3d --local-dir dataset/organmnist3d --repo-type dataset
```
Then run the `quick_start.sh` to test (It may take 1-2 hours, and make sure your GPU memory is over 12 GB):
```
bash quick_start.sh
```

# 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects, our code is developed based on their contributions:
     [aideml](https://github.com/WecoAI/aideml) [monai](https://github.com/Project-MONAI) [AutoGluon](https://github.com/autogluon/autogluon) [AutoKeras](https://github.com/keras-team/autokeras) [Auto-SKlearn](https://github.com/automl/auto-sklearn) [ML-Master](https://github.com/sjtu-sai-agents/ML-Master)
