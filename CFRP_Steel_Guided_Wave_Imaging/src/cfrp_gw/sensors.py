from __future__ import annotations

import numpy as np
import pandas as pd


def local_receiver_offsets_mm() -> np.ndarray:
    """Return the 16 receiver offsets used around each transmitter location."""
    levels = (-50.0, -25.0, 0.0, 25.0, 50.0)
    points = [
        (dx, dy)
        for dx in levels
        for dy in levels
        if max(abs(dx), abs(dy)) == 50.0
    ]
    return np.asarray(points, dtype=float)


def manuscript_sensor_layout() -> pd.DataFrame:
    """Return the 17 x 16 transmitter-receiver layout described in Section 3.4."""
    offsets = local_receiver_offsets_mm()
    rows = []
    path_id = 0
    for tx_x in np.arange(50.0, 851.0, 50.0):
        tx_y = 50.0
        for receiver_id, (dx, dy) in enumerate(offsets, start=1):
            rows.append(
                {
                    "Path_ID": path_id,
                    "Tx_X": tx_x,
                    "Tx_Y": tx_y,
                    "Rx_X": tx_x + dx,
                    "Rx_Y": tx_y + dy,
                    "Receiver_ID": receiver_id,
                }
            )
            path_id += 1
    return pd.DataFrame(rows)
