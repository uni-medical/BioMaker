# Desktop engine dependency policy

Desktop packages use the same `biomaker` source and core requirements as the CLI
and WebUI. This directory contains generated copies of the root requirements and lock.
The package gate requires byte-for-byte equality; change dependency policy at
the root, then regenerate these copies.

Run `npm run lock:engine -- --upgrade` after reviewing requirement changes, then
regenerate the portable bundle and verify real engine installation and imports.
`source.json` binds the overlay to the root manifests; packaging rejects stale
metadata. Each installed environment retains its lock and versioned engine path.
The package gate also checks the bundled engine checksum and compares every
allowed source file with the current repository; rebuild the runtime after
changing core or WebUI source, even when dependency manifests are unchanged.

A successful import, dependency audit or synthetic workflow verifies only its
stated software scope. It does not establish scientific equivalence between
runtime versions or validate models produced by previous experiments.
