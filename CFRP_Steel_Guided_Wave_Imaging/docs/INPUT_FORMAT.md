# Input formats

No experimental data are distributed with this repository.

## Path-energy table

The imaging scripts accept CSV or XLSX files containing one row per transmitter-receiver path. Required columns are:

- `Tx_X`, `Tx_Y`: transmitter coordinates in mm
- `Rx_X`, `Rx_Y`: receiver coordinates in mm
- `Energy`: median path energy after the fixed-band/Hilbert preprocessing

As an alternative to `Energy`, the loader accepts `Energy_mean`, or five repeat-energy columns named `Energy_1` through `Energy_5`; in the latter case the median is calculated automatically.

The manuscript configuration contains 17 transmitter positions and 16 local receiver positions, giving 272 paths per scan. The code does not hard-code the row order; it uses the coordinates in each row.

## Waveform NPZ input

`scripts/preprocess_waveforms.py` provides the Appendix-A preprocessing chain for waveform arrays. The input NPZ file contains:

- `waveforms`: array with shape `(n_paths, 5, n_samples)`
- `tx_x`, `tx_y`, `rx_x`, `rx_y`: one-dimensional coordinate arrays of length `n_paths`

At the manuscript sampling rate of 2 MS/s, each trace must contain at least 2000 samples so that the effective 1.0 ms record can be processed.

The acquisition system stored the measurements as TDMS files. TDMS group/channel organization is instrument- and campaign-specific and is therefore not hard-coded here. Exporting each repeated path waveform into the array layout above preserves the filtering, Hilbert-envelope, energy-integration, and median-statistic steps described in Appendix A.
