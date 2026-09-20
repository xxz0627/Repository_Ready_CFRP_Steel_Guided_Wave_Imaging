#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cfrp_gw.config import DEFAULT_CONFIG
from cfrp_gw.io import read_table
from cfrp_gw.pipeline import run_imaging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the parameter sweeps reported in the manuscript.")
    parser.add_argument("input", help="CSV/XLSX path-energy table")
    parser.add_argument("--defect-mode", choices=["single", "multiple"], default="single")
    parser.add_argument("--output", default="outputs/sensitivity_sweep.json")
    args = parser.parse_args()

    table = read_table(args.input)
    results = {}

    sweeps = {
        "tv_weight": [0.0, 0.001, 0.003, 0.006],
        "high_peak_ratio": [0.40, 0.45, 0.50, 0.55],
        "iterations": [50, 100, 150],
        "path_mad_k": [0.5, 1.0, 1.5],
        "ssb_percentile": [0.50, 0.60, 0.70, 0.80],
    }

    for parameter, values in sweeps.items():
        parameter_results = []
        for value in values:
            cfg = DEFAULT_CONFIG
            if parameter == "tv_weight":
                cfg = replace(cfg, reconstruction=replace(cfg.reconstruction, tv_weight=value))
            elif parameter == "high_peak_ratio":
                cfg = replace(cfg, segmentation=replace(cfg.segmentation, high_peak_ratio=value))
            elif parameter == "iterations":
                cfg = replace(cfg, reconstruction=replace(cfg.reconstruction, iterations=value))
            elif parameter == "path_mad_k":
                cfg = replace(cfg, ssb=replace(cfg.ssb, mad_k=value))
            elif parameter == "ssb_percentile":
                cfg = replace(cfg, ssb=replace(cfg.ssb, percentile=value))

            method = "SIRT" if parameter == "tv_weight" and value == 0.0 else "TV-SIRT"
            result = run_imaging(table, method=method, defect_mode=args.defect_mode, config=cfg)
            parameter_results.append(
                {
                    "value": value,
                    "method": method,
                    "estimates": [item.to_dict() for item in result.estimates],
                    "nonzero_path_observations": int((result.paths["Attenuation"] > 0).sum()),
                }
            )
        results[parameter] = parameter_results

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
