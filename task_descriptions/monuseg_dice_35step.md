# MoNuSeg Binary Segmentation

Develop a robust nuclei foreground segmentation model for MoNuSeg.

Use only the agent-visible training split for training and the agent-visible validation split for model selection. Generate official test predictions in `./working/submission.csv`.

The expected output is a foreground mask for each test image. Background should be 0. Nucleus foreground can be any positive value.

Primary validation metric: Binary Dice, treating every nonzero pixel as foreground.

For the final user-only official test report, use Binary Dice first. AJI may also be reported privately if instance masks are produced, but it is not the optimization target for this run.
