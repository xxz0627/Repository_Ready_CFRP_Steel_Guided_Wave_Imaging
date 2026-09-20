# CFRP-Steel Guided-Wave Attenuation Imaging

Source code for the processing and imaging workflow described in:

**Historical-Baseline-Free Guided-Wave Imaging of CFRP-Steel Interfacial Debonding with a Movable Magnetic PZT Array**

The implementation follows the manuscript's historical-baseline-free formulation. Path energies from the current scan are grouped by nominal propagation length, the 70th percentile within each group is used as the same-scan statistical-background (SSB) reference, and the resulting log-ratio attenuation observations are screened with the path-domain MAD rule before reconstruction.

The repository contains code only. Experimental waveform and path-energy data are not distributed here; they are available from the corresponding author upon reasonable request.

## Processing chain

1. Demean each repeated waveform.
2. Apply a zero-phase 80-120 kHz Butterworth band-pass filter.
3. Form the Hilbert envelope and integrate envelope-squared energy over the effective 1.0 ms record.
4. Use the median energy of five repeated acquisitions for each path.
5. Construct the 70th-percentile SSB reference separately for the nominal 50.0, 55.9, and 70.7 mm path-length groups.
6. Apply the path-domain MAD screen (`k = 1.0`, threshold cap `0.04`).
7. Reconstruct the 160 x 24 equivalent attenuation field using SIRT or TV-SIRT (100 iterations, `lambda_TV = 0.003`).
8. Segment the reconstructed field using the image-domain robust threshold and quantify single or multiple defect-response regions.

A PDI implementation with the fixed shape factor `beta = 1.04` is included for the comparison described in Section 4.4.1. The parameter-sweep script covers the TV weight, segmentation high-threshold ratio, SIRT iteration count, path-domain MAD coefficient, and SSB percentile examined in Section 4.4.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Input

The main scripts use path-energy tables with transmitter/receiver coordinates and either a median `Energy` column or five repeated energy columns. The expected formats are documented in `docs/INPUT_FORMAT.md`.

No experimental data are included in the repository.

## Run TV-SIRT

```bash
python scripts/run_imaging.py path_energy.xlsx --method TV-SIRT --defect-mode single
```

For the four-response A4 configuration:

```bash
python scripts/run_imaging.py path_energy_A4.xlsx --method TV-SIRT --defect-mode multiple
```

## Compare PDI, SIRT, and TV-SIRT

```bash
python scripts/compare_methods.py path_energy.xlsx --defect-mode single
```

## Parameter sweeps

```bash
python scripts/sensitivity_sweep.py path_energy.xlsx --defect-mode single
```

The script writes detected centroids and equivalent-area estimates for each parameter setting. Numerical values reported in the manuscript require the experimental data, which are not included in this repository.

## Waveform preprocessing

If repeated waveforms have been exported from the acquisition files into an NPZ array, the Appendix-A preprocessing chain can be run with:

```bash
python scripts/preprocess_waveforms.py repeated_waveforms.npz path_energy.csv
```

The NPZ layout is documented in `docs/INPUT_FORMAT.md`.

## Output

`run_imaging.py` writes:

- screened SSB path observations as CSV;
- the reconstructed attenuation map as PNG;
- detected centroid and equivalent-area estimates as JSON.

Quantification is performed on the reconstructed field before display interpolation. The image interpolation is used for visualization only.

## Data availability

Experimental data are available from the corresponding author upon reasonable request. See `docs/DATA_ACCESS.md`.

## Citation

Citation metadata are provided in `CITATION.cff`.
