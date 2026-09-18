# CFRP–Steel Interfacial Debonding Guided-Wave Imaging

GitHub-ready research-code package accompanying the manuscript **“Historical-Baseline-Free Guided-Wave Imaging of CFRP-Steel Interfacial Debonding with a Movable Magnetic PZT Array.”**

## Contents

- `src/` – reconstruction, TV-SIRT, quantification, visualization, and data-loading modules supplied by the authors.
- `src/ssb_reference.py` – compact same-scan statistical-background (SSB) construction matching the manuscript definition around Eqs. (15)–(17).
- `scripts/compute_ssb_observations.py` – command-line helper that constructs the 70th-percentile, path-length-group SSB observations and applies the path-domain MAD screen.
- `data/processed/` – representative processed path-energy tables for the healthy state and A1–A4 cases.
- `docs/MANUSCRIPT_MAPPING.md` – mapping between manuscript sections and repository files.

Raw TDMS records are **not** included in this compact public-release package; the representative processed energy tables are supplied so that the reconstruction/quantification stages can be inspected and reused without distributing the much larger raw acquisition archive.

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Same-scan SSB observation check

The manuscript groups paths by nominal propagation length and uses the 70th percentile of the **current scan** as the internal energy reference within each group. The helper below creates these path-level observations from one processed energy table:

```bash
python scripts/compute_ssb_observations.py data/processed/damage_A1.xlsx
```

The resulting CSV is written to `outputs/` and includes the path-length group, SSB reference energy, raw log-ratio attenuation observation, MAD threshold, and screened observation.

## Reconstruction code

The authors' current modular reconstruction implementation is retained under `src/`. To run the supplied SIRT/TV-SIRT comparison scripts, execute them from `src/` after checking the analysis settings in `src/config.py`:

```bash
cd src
python run_two_methods_all_data.py
```

The repository intentionally keeps analysis parameters explicit in `config.py` so that the configuration used for a released result can be archived with the code. Before creating the permanent GitHub/Zenodo release, the authors should run the public-release checklist and archive the exact final configuration together with the manuscript version.

## Data columns

The processed spreadsheets contain transmitter/receiver coordinates and path-energy statistics used by the analysis. Coordinate columns are `Tx_X`, `Tx_Y`, `Rx_X`, and `Rx_Y`; the energy column is read as `Energy_mean` when available, otherwise `Energy`.

## Citation

Please cite the associated manuscript. A `CITATION.cff` file is included for GitHub citation metadata.

## License

No software license has been assigned automatically. The authors should select and add the intended open-source license before making the repository public.
