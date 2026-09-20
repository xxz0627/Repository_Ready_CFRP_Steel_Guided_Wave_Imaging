#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cfrp_gw.config import DEFAULT_CONFIG
from cfrp_gw.preprocessing import repeated_path_energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert repeated waveform arrays to path-energy features using the manuscript preprocessing chain."
    )
    parser.add_argument("input", help="NPZ file described in docs/INPUT_FORMAT.md")
    parser.add_argument("output", help="output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    content = np.load(args.input)
    waveforms = np.asarray(content["waveforms"], dtype=float)
    if waveforms.ndim != 3:
        raise ValueError("waveforms must have shape (n_paths, 5, n_samples)")

    required = ["tx_x", "tx_y", "rx_x", "rx_y"]
    for key in required:
        if key not in content:
            raise ValueError(f"missing NPZ array: {key}")
    if any(len(content[key]) != waveforms.shape[0] for key in required):
        raise ValueError("coordinate array lengths must match the number of waveform paths")

    energies = []
    repeat_energies = []
    for path_traces in waveforms:
        median_energy, individual = repeated_path_energy(path_traces, DEFAULT_CONFIG.preprocessing)
        energies.append(median_energy)
        repeat_energies.append(individual)

    rows = {
        "Tx_X": content["tx_x"],
        "Tx_Y": content["tx_y"],
        "Rx_X": content["rx_x"],
        "Rx_Y": content["rx_y"],
        "Energy": energies,
    }
    repeat_energies = np.asarray(repeat_energies)
    for index in range(repeat_energies.shape[1]):
        rows[f"Energy_{index + 1}"] = repeat_energies[:, index]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
