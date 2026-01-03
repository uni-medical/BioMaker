# CREA

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
   uv pip install -r requirements.txt
   uv pip install -e .
   ```

4. **Set up API keys**:
   Set up your private API key for OPENAI-compatible endpoints:
   ```bash
   export OPENAI_BASE_URL="<your base url>" # (e.g. https://api.openai.com/v1)
   export OPENAI_API_KEY="<your key>"
   ```
## Setup knowledge base for RAG Tool
You can create the knowledge base with `create_rag_db.sh`.

You could use this command to check whether the creation succeeds.
```
python -m meddle.monai_rag.query_rag_db
``` 
## Test with Example Dataset
You can download the example dataset from [Huggingface](https://huggingface.co/datasets/blueyo0/organmnist3d).
Download command:
```
huggingface-cli download blueyo0/organmnist3d --local-dir organmnist3d
```
Then run the `quick_start.sh` to test:
```
meddle data_dir="organmnist3d" exp_name="test-organmnist3d" \
     goal="Predict the category of each given 3D Abdominal CT image by selecting the most appropriate one from the 11 available classes." \
     eval="Use the accuracy and the area under curve (AUC) between the predicted class and ground-truth class on the test set."
```

# 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects, our code is developed based on their contributions:
     - [aideml](https://github.com/WecoAI/aideml)
     - [monai](https://github.com/Project-MONAI)
     - [AutoGluon](https://github.com/autogluon/autogluon)
     - [AutoKeras](https://github.com/keras-team/autokeras)
     - [Auto-SKlearn](https://github.com/automl/auto-sklearn)
     - [ML-Master](https://github.com/sjtu-sai-agents/ML-Master)
