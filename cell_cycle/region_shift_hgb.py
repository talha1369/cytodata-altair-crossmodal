#!/usr/bin/env python
"""Where on the 4i clock do knockouts move their cells? (HistGB library, ko_library_hgb/cell_predictions.npz)
Light: per-cell predicted phase for 52,189 NTC (out-of-fold) and 591,136 knockout cells.
Clock regions use the corrected labels (cc_phase 0..1, wraps):
  0-0.25 G1 (2N) | 0.25-0.375 late G1 entering S | 0.375-0.5 S | 0.5-0.625 G2 (4N) | 0.625-0.75 mitosis/division |
  0.75-1 G1 after division
Writes ko_library_hgb/region_shift_hgb.csv (per KO: fraction of cells in each region minus the NTC fraction, in % points).
usage: python region_shift_hgb.py [HGB_DIR]"""
import sys
import numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "ko_library_hgb"
EDGES = [0, .25, .375, .5, .625, .75, 1.0001]
NAMES = ["G1 (0-0.25)", "late G1 entering S (0.25-0.375)", "S (0.375-0.5)", "G2 (0.5-0.625)",
         "mitosis/division (0.625-0.75)", "G1 after division (0.75-1)"]
f = np.load(f"{D}/cell_predictions.npz", allow_pickle=True)
def frac(p): return np.histogram(np.mod(p, 1), bins=EDGES)[0] / len(p)
ntc = frac(f["ntc_phase_oof"])
print("NTC predicted region fractions (n=52,189, out-of-fold): " + ", ".join(f"{n} {100 * v:.1f}%" for n, v in zip(NAMES, ntc)))
g = pd.Series(f["ko_gene"].astype(str)); ph = f["ko_phase"]
rows = {k: 100 * (frac(ph[ix.values]) - ntc) for k, ix in g.groupby(g).groups.items()}
R = pd.DataFrame(rows, index=NAMES).T; R.index.name = "gene"
R["n"] = g.value_counts().reindex(R.index).values
lib = pd.read_csv(f"{D}/ko_library_cycle.csv").set_index("gene")
R = R.join(lib[["q_cycle", "cycle_TV", "d_pct_4N_pred", "measured_pRb_shift_log2"]])
R.to_csv(f"{D}/region_shift_hgb.csv")
pd.set_option("display.width", 250)
foc = ["CDK7", "CCNB1", "POLA1", "CDC45", "MCM7", "RRM1", "MTBP", "KIF11", "RPS6"]
print("\nfocus knockouts, change in % of cells per clock region vs NTC (HistGB):\n" + R.reindex(foc).round(2).to_string())
sig = R[R.q_cycle < .05]
print(f"\namong {len(sig)} q_cycle<0.05 knockouts: median change per region (% points)\n" + sig[NAMES].median().round(2).to_string())
print("\nregion with the largest GAIN, count of significant KOs:\n" + sig[NAMES].idxmax(axis=1).value_counts().to_string())
print("\nregion with the largest LOSS, count of significant KOs:\n" + sig[NAMES].idxmin(axis=1).value_counts().to_string())
up, dn = sig[sig.d_pct_4N_pred > 0], sig[sig.d_pct_4N_pred < 0]
print(f"\nsignificant KOs with predicted %4N UP (n={len(up)}): median region change\n" + up[NAMES].median().round(2).to_string())
print(f"\nsignificant KOs with predicted %4N DOWN (n={len(dn)}): median region change\n" + dn[NAMES].median().round(2).to_string())
print("DONE")
