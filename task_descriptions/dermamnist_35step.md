# DermAMNIST Classification

Develop a robust medical image classification model for DermAMNIST.

Use only the agent-visible training split for training and the agent-visible validation split for model selection. Generate official test predictions in `./working/submission.csv`.

Primary validation metric: macro one-vs-rest ROC AUC.
Secondary validation metric: accuracy.

For the final user-only official test report, use Accuracy and macro one-vs-rest AUC.
