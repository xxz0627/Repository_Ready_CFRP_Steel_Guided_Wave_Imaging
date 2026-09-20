#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfrp_gw.config import DEFAULT_CONFIG
from cfrp_gw.display import save_attenuation_map
from cfrp_gw.io import read_table
from cfrp_gw.pipeline import run_imaging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SSB-based CFRP-steel guided-wave attenuation imaging.")
    parser.add_argument("input", help="CSV/XLSX path-energy table")
    parser.add_argument("--method", choices=["TV-SIRT", "SIRT", "PDI"], default="TV-SIRT")
    parser.add_argument("--defect-mode", choices=["single", "multiple", "none"], default="single")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = read_table(args.input)
    result = run_imaging(table, method=args.method, defect_mode=args.defect_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    method_tag = args.method.lower().replace("-", "_")

    result.paths.to_csv(out_dir / f"{stem}_{method_tag}_path_observations.csv", index=False)
    save_attenuation_map(
        result.field,
        result.grid,
        DEFAULT_CONFIG.reconstruction,
        out_dir / f"{stem}_{method_tag}_attenuation.png",
        title=f"{args.method} attenuation map",
    )
    estimates = [estimate.to_dict() for estimate in result.estimates]
    with (out_dir / f"{stem}_{method_tag}_quantification.json").open("w", encoding="utf-8") as handle:
        json.dump(estimates, handle, indent=2)

    print(f"method: {result.method}")
    print(f"paths: {len(result.paths)}")
    print(f"nonzero screened observations: {(result.paths['Attenuation'] > 0).sum()}")
    print(f"detected regions: {len(result.estimates)}")
    print(f"outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
