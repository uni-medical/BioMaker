# ATM22 reference generation

`tree_parse.py` is an unmodified copy of the official ATM22 related-work
repository's `evaluation/tree_parse.py`:

- Repository: https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work
- Commit: `522081149dfde905192046bd7237b8ace7819600`
- Source: https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/blob/522081149dfde905192046bd7237b8ace7819600/evaluation/tree_parse.py
- SHA256: `1fce3c18073e03c398b471100f3867d025a33704a7654e6b40d825302c753099`
- License: MIT, Copyright (c) 2022 Puzzled-Hui; the accompanying `LICENSE`
  is copied verbatim from the same commit.

BioMaker's `airway_reference.py` calls these functions in the original driver's
order, including the mutating `whether_refinement` call. It replaces only the
hardcoded demonstration file paths and geometry-losing NIfTI writes with checked
volume IO and a content-addressed cache. The graph serialization and strict
validation are BioMaker adapters, not changes to the upstream algorithm.

This is the **local `atm22refs-v1` protocol**, not a claim that ATM26's hidden
reference structures were produced by this procedure. Original anatomical labels
are never rewritten. This builder runs separately during reference preparation;
frozen scoring must consume existing references and must not run it for every
prediction.

The original API requires `scikit-image==0.22.0`; version 0.24 changes its output to boolean, which makes the unchanged
upstream neighbor-count convolution boolean and collapses real bifurcations.
Version 0.22 preserves the original uint8 0/1 behavior for uint8 0/1 input;
version 0.25 removes the function entirely. No compatibility monkeypatch or alternate skeleton algorithm
is applied. The cache records exact dependency versions, source hashes, original
label digest and geometry, and hashes of every generated artifact.
