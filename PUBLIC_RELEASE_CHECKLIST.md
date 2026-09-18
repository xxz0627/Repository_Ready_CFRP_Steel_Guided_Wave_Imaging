# Public-release checklist

1. Create the GitHub repository and upload this folder **without adding any bundled font files**.
2. Select and add the intended software license.
3. Create a clean Python environment and run `pip install -r requirements.txt`.
4. Run `python scripts/compute_ssb_observations.py data/processed/damage_A1.xlsx`.
5. Run the intended reconstruction entry point and record the final `src/config.py` used for the archived release.
6. Confirm that no private raw data, credentials, local absolute paths, or temporary output files are included.
7. Insert the GitHub URL into the manuscript and all response documents where `[GITHUB URL TO BE INSERTED]` appears.
8. Remove the red author-only reminder from the Abstract before final submission.
9. Optionally archive the GitHub release with Zenodo and add the DOI to `CITATION.cff` and Code Availability.
