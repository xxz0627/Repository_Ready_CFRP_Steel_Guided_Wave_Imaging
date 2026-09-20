# Manuscript-to-code mapping

The repository follows the processing chain described in **Historical-Baseline-Free Guided-Wave Imaging of CFRP-Steel Interfacial Debonding with a Movable Magnetic PZT Array**.

| Manuscript item | Implementation |
|---|---|
| Section 2.1 / Appendix A, Eqs. (1)-(12): DC removal, fixed 80-120 kHz filtering, Hilbert envelope, 1.0 ms energy integration, five-repeat median | `src/cfrp_gw/preprocessing.py` |
| Section 3.4: 17 transmitter positions and 16 local receiver positions (272 paths) | `src/cfrp_gw/sensors.py` |
| Eqs. (15)-(17): 70th-percentile same-scan statistical background within nominal path-length groups and path-domain MAD screening | `src/cfrp_gw/ssb.py` |
| Eqs. (18)-(20): straight-ray path matrix on the 160 x 24 grid with 200 path samples | `src/cfrp_gw/geometry.py` |
| Eqs. (21)-(22): SIRT and TV-SIRT | `src/cfrp_gw/reconstruction.py` |
| Section 2.2.3: display interpolation and independent panel normalization | `src/cfrp_gw/display.py` |
| Eqs. (24)-(32): image-domain robust thresholding and single-response equivalent-area estimate | `src/cfrp_gw/quantification.py` |
| Eq. (33): multiple-response profile/soft-profile area fusion | `src/cfrp_gw/quantification.py` |
| Section 4.4.1: PDI baseline with fixed shape factor beta = 1.04 | `src/cfrp_gw/pdi.py` |
| PDI / SIRT / TV-SIRT comparison using the same path observations | `scripts/compare_methods.py` |
| Section 4.4 parameter checks | `scripts/sensitivity_sweep.py` |
| Section 4.4.2 path-coverage diagnostics | `src/cfrp_gw/diagnostics.py` |

The main imaging pipeline uses only the current scan to construct the SSB reference. It does not require a separately recorded healthy-state baseline.

Reference defect coordinates and areas are intentionally absent from the detection pipeline. If ground-truth geometry is supplied for a controlled experiment, `evaluate_estimates()` can be used after detection to calculate centroid offset and equivalent-area error.
