from pathlib import Path
import sys
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from ssb_reference import construct_ssb_observations, mad_screen

def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/compute_ssb_observations.py data/processed/damage_A1.xlsx [output.csv]')
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO/'outputs'/(inp.stem+'_ssb_observations.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(inp)
    ssb = construct_ssb_observations(df, q=0.70)
    screened, thr = mad_screen(ssb['Raw_Attenuation_Integral'].to_numpy(), k=1.0, max_threshold=0.04)
    ssb['Travel_Time_us'] = screened
    ssb['Denoise_Threshold'] = thr
    ssb.to_csv(out, index=False)
    print(f'Wrote {out}')
    print(f'MAD threshold = {thr:.6f}; nonzero paths = {(screened>0).sum()}/{len(screened)}')

if __name__ == '__main__':
    main()
