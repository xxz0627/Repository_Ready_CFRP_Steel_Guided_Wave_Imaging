from cfrp_gw.config import DEFAULT_CONFIG


def test_manuscript_default_parameters():
    cfg = DEFAULT_CONFIG
    assert cfg.preprocessing.sampling_rate_hz == 2_000_000.0
    assert cfg.preprocessing.low_cut_hz == 80_000.0
    assert cfg.preprocessing.high_cut_hz == 120_000.0
    assert cfg.preprocessing.repeats_per_path == 5
    assert cfg.ssb.percentile == 0.70
    assert cfg.ssb.mad_k == 1.0
    assert cfg.ssb.mad_threshold_cap == 0.04
    assert cfg.grid.nx == 160 and cfg.grid.ny == 24
    assert cfg.grid.ray_samples == 200
    assert cfg.reconstruction.iterations == 100
    assert cfg.reconstruction.tv_weight == 0.003
    assert cfg.segmentation.low_peak_ratio == 0.18
    assert cfg.segmentation.high_peak_ratio == 0.45
    assert cfg.pdi.shape_factor_beta == 1.04
