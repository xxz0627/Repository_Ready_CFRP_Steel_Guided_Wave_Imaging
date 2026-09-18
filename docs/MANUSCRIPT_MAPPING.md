# Manuscript-to-code mapping

| Manuscript item | Repository implementation |
|---|---|
| Section 2.1 / Appendix A: standard preprocessing definitions | The repository starts from representative processed energy tables; Appendix A provides the mathematical preprocessing derivations. |
| Eqs. (15)–(17): same-scan SSB reference and path-domain screening | `src/ssb_reference.py`, `scripts/compute_ssb_observations.py` |
| Eqs. (18)–(22): SIRT / TV-SIRT reconstruction | `src/reconstructor.py`, `src/ray_tracer.py`, `src/grid_manager.py` |
| Eqs. (24)–(33): segmentation and equivalent-area quantification | `src/quantifier.py`, settings in `src/config.py` |
| Figures / display normalization | `src/visualizer.py` |
| Conventional SIRT vs TV-SIRT run | `src/run_two_methods_all_data.py` |

## Release note

This folder is prepared as a clean GitHub upload package. Replace every manuscript placeholder `[GITHUB URL TO BE INSERTED]` with the final repository URL after the repository is created. If a DOI is minted (e.g., via Zenodo), add it to the manuscript Code Availability statement and `CITATION.cff`.
