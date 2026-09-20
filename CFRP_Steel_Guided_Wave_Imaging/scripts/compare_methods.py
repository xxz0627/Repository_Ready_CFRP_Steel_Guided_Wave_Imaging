#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfrp_gw.display import save_attenuation_map
from cfrp_gw.io import read_table
from cfrp_gw.pipeline import run_imaging
from cfrp_gw.config import DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDI, SIRT, and TV-SIRT on the same SSB path observations.")
    parser.add_argument("input", help="CSV/XLSX path-energy table")
    parser.add_argument("--defect-mode", choices=["single", "multiple", "none"], default="single")
    parser.add_argument("--output-dir", default="outputs/method_comparison")
    args = parser.parse_args()

    table = read_table(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    summary = {}
    for method in ("PDI", "SIRT", "TV-SIRT"):
        result = run_imaging(table, method=method, defect_mode=args.defect_mode)
        tag = method.lower().replace("-", "_")
        save_attenuation_map(
            result.field,
            result.grid,
            DEFAULT_CONFIG.reconstruction,
            out_dir / f"{stem}_{tag}.png",
            title=f"{method} attenuation map",
        )
        summary[method] = [item.to_dict() for item in result.estimates]

    with (out_dir / f"{stem}_method_quantification.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
