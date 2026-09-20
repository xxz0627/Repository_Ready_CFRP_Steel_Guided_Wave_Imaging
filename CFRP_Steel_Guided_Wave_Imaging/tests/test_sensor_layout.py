import numpy as np

from cfrp_gw.sensors import manuscript_sensor_layout


def test_manuscript_sensor_layout_has_272_paths_and_three_lengths():
    layout = manuscript_sensor_layout()
    assert len(layout) == 272
    assert layout["Tx_X"].nunique() == 17
    lengths = np.hypot(layout["Rx_X"] - layout["Tx_X"], layout["Rx_Y"] - layout["Tx_Y"])
    unique = np.unique(np.round(lengths, 1))
    assert np.array_equal(unique, np.array([50.0, 55.9, 70.7]))
    assert layout["Rx_X"].between(0, 900).all()
    assert layout["Rx_Y"].between(0, 100).all()
