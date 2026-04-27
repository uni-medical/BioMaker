<div align="center" id="biomakertop">
<img src="./assets/BioMaker_logo.png" alt="logo" width="400" margin="10px"></img>
<br/>
<a href="https://pypi.org/project/biomaker"><img src="https://img.shields.io/pypi/v/biomaker" alt="PyPI"></a>
<a href="https://github.com/Ziyan-Huang/BioMaker/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="license"></a>
<img src="https://img.shields.io/badge/Python-3.11-A7D8FF" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.x-FCD299" alt="PyTorch">
<img src="https://img.shields.io/badge/Paper-to%20be%20released-b31b1b?logo=arxiv&logoColor=red" alt="Paper">
</div>

# Introduction
**BioMaker** is an agentic development system for biomedical image analysis.

BioMaker uses a compact workflow: it optionally validates a user-provided solution file, then runs a single proposal agent loop that drafts code, debugs failed attempts, and improves successful solutions. The workflow can use a lightweight MONAI knowledge index and web search when extra biomedical implementation context is useful.

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

2. **Clone the repository**:
     ```bash
     git clone https://github.com/Ziyan-Huang/BioMaker
     cd BioMaker
     ```

3. **Create and sync a Python 3.11 environment**:
     ```bash
     uv sync --python 3.11
     ```

4. **Set up API keys**:
Set up your private API key for OpenAI-compatible endpoints. Serper and Jina are optional but recommended for web search: Serper finds relevant pages, and Jina reads those pages into cleaner text.
     ```bash
     export OPENAI_BASE_URL="<your base url>" # (e.g. https://api.openai.com/v1)
     export OPENAI_API_KEY="<your key>"
     export SERPER_API_KEY="<your serper key>"
     export JINA_API_KEY="<your jina key>"
     ```
If `SERPER_API_KEY` is missing, BioMaker warns and disables web search for that run. If `JINA_API_KEY` is missing, BioMaker warns and falls back to Serper snippets only.

5. **Set up the MONAI RAG index**
BioMaker keeps the RAG index local instead of storing generated index files in git. Build it once before running tasks:
     ```bash
     bash scripts/create_rag_db.sh
     ```

     Check whether the index is usable:
     ```bash
     uv run python -m biomaker.monai_rag.query_rag_db
     ```
If the index is missing, BioMaker warns and disables MONAI RAG for that run.

## Workflow

The default run is intentionally simple:

```text
prepare workspace and data
-> ProposalAgent loop
   -> draft a solution when no good candidate exists
   -> debug failed candidates
   -> improve the best successful candidate
-> save journal, tree, best solution, and optional final report
```

There is no AutoGluon, Optuna, or separate HPO stage in the default workflow. Increase `agent.steps` for longer searches, or lower it for quick local runs.

The proposal loop is a lightweight solution tree search. BioMaker drafts a conservative baseline, debugs failed leaf nodes until a runnable solution exists, then mostly improves top-performing nodes. It can still create extra draft roots later to preserve diversity.

Generated code runs under a strict dependency policy. BioMaker tells the model which installed packages are allowed, blocks package installation, and performs a preflight import check before execution. If code imports an unavailable package such as `cv2` or `albumentations`, the run fails fast with a policy error so the next debug step can replace it with an allowed package.

## Test with Example Dataset
You can download the example dataset from [Huggingface](https://huggingface.co/datasets/blueyo0/organmnist3d).
Download command:
```
uv run hf download blueyo0/organmnist3d --local-dir dataset/organmnist3d --repo-type dataset
```
Then run the `quick_start.sh` to test (It may take 1-2 hours, and make sure your GPU memory is over 12 GB):
```
bash quick_start.sh
```

# 🙏 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source projects that make biomedical AI development easier, especially [MONAI](https://github.com/Project-MONAI) and [aideml](https://github.com/WecoAI/aideml).
