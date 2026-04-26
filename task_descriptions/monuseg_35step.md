# MoNuSeg Instance Segmentation

Develop a robust nuclei instance segmentation model for MoNuSeg.

Use only the agent-visible training split for training and the agent-visible validation split for model selection. Generate official test predictions in `./working/submission.csv`.

The expected output is an instance mask for each test image. Each nucleus instance should have a distinct positive integer label. Background should be 0.

Primary validation metric: AJI.
Secondary validation metric: Binary Dice, treating every nonzero pixel as foreground.

For the final user-only official test report, use AJI and Binary Dice.
