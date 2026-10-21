# Resolution Reduction in Phase-Congruency Template Matching

Code and numerical evidence for **Resolution Reduction in Phase-Congruency Template Matching: Computational Savings and Localisation Limits**.

Package revision: 2026-10-21. Experimental observations and computation scripts retain their frozen versions; this package revision updates delivery documentation and excludes internal working material.

## Contents

- `code/`: scalar feature construction, matching, experiment runner, statistical analysis, meaningful numerical tests and dependency versions.
- `results/`: the 122-pair selection, preprocessing metadata, primary/control observations, repeated timings and complete statistical tables.
- `experiments/python_baselines/`: HOPC-Py, simplified RIFT-Py, validation, frozen protocol, 8,052 observations and 900 timing rows.
- `experiments/speckle_stress/`: multiplicative perturbation implementation, protocol, 54,600 observations and analysis.
- `notes/source_data_dictionary_20261018.txt`: data fields, sampling units and reproduction boundaries. The filename is retained for stable references; delivery notes were updated on 2026-10-21.
- `archive_manifest.json`: relative file paths, sizes and SHA-256 hashes, excluding the manifest itself and `no_public/`.
- `no_public/`: internal planning, validation records, manuscript drafts and figure source tables. The folder is excluded by `.gitignore` and is not part of the public release.

## Reproduction

Start with `code/README.md`. Numerical reanalysis of saved observations does not require the provider images. Full image matching requires lawful access to QXS-SAROPT and ISPRS Potsdam and preparation of the selected image crops. Do not overwrite the frozen results when starting a new execution.

Figure source tables are not distributed in this package. All reported figure values are derived from the observations and statistics supplied under `results/` and `experiments/`; the rendered map panels additionally require the provider images.

All analyses are exploratory. The 122 selected pairs are analysis units, not 122 verified independent scenes. Six offsets are repeated measurements within each pair. SAR parent-scene grouping is unavailable. HOPC-Py and RIFT-Py are disclosed Python implementations; validation does not establish equality to original-author MATLAB software or to full rotation-invariant RIFT.

## Third-party materials and licensing

Original images, pixel caches, downloaded papers and upstream native binaries are excluded. Obtain images from the providers cited in the manuscript. See `experiments/python_baselines/THIRD_PARTY_NOTICES.txt` for algorithm-port notices. No new project-wide software licence has been assigned in this packaging step; the authors should choose suitable terms for their own code when publishing, while preserving existing third-party notices.

For public release, host-specific interpreter and input paths in the delivered protocols and derived summaries were replaced by neutral placeholders, and the affected run signatures were re-derived from the redacted content. Version, platform and library metadata are retained. The setup and reproduction instructions in `code/README.md` use portable commands.
