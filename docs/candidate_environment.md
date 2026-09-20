# Candidate environment

BioMaker provides a toolbox, not a prescribed modeling method. Generated
solutions may choose any installed package listed in the environment manifest;
they do not have to use any particular library. Installing a dependency does not
require a candidate to import it or replace an existing implementation.

`requirements.txt` and `uv.lock` provision the toolbox. `PACKAGE_SPECS` in
`biomaker/utils/environment.py` controls its agent-facing names and installed
version reporting. Keep all three consistent when extending the toolbox.

The available utilities include einops, tqdm, networkx, h5py and TensorBoard.
The medical toolbox additionally includes MONAI, SimpleITK, nibabel, pydicom,
scikit-image, cc3d and TorchIO. Candidate use of these libraries does not alter
Harness-owned scoring formulas or reference generation.

Before executing a candidate, `validate_code_environment` distinguishes:

- A declared required import whose module is missing: report "allowed package
  not installed" instead of pretending it is available.
- An undeclared import: report the unsupported dependency and available choices.
- An optional declared import guarded by an import-error handler: allow the
  candidate's fallback when the dependency is absent.

This is an early dependency check, not proof that every symbol or API call is
valid. A nonexistent symbol in an installed library (for example an incorrect
MONAI import) still raises an execution error for the agent to debug. API probes
and execution feedback remain necessary. Candidate package-manager commands stay
disabled so concurrent candidates cannot mutate each other's environment.

Environment changes do not update manifests already captured by running agents.
Use the updated manifest on the next process start; do not restart healthy
experiments merely to advertise newly installed optional tools.
