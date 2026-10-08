# Desktop engine dependency policy

The research checkout retains its original dependency constraints and lock file.
Desktop packages copy the same engine source but overlay **requirements.txt** and
**uv.lock** from this directory. This lets a new desktop installation use patched
libraries without silently changing historical research environments.

The initial desktop lock updates Requests, MONAI and PyTorch, plus transitive
packages with known advisories at build time. A successful desktop import or
dependency audit does not establish scientific equivalence with a historical
experiment. Rerun scientific validation before comparing model results across
runtime versions. Every installed environment retains its own exact lock and
versioned engine path.

Run `npm run lock:engine -- --upgrade` after reviewing requirement changes, then
regenerate the portable bundle and perform real engine installation/import tests.
`source.json` records the base research manifests used for the overlay; packaging
rejects stale metadata when those manifests change.
