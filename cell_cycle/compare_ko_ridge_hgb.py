#!/usr/bin/env python
"""Compare the knockout cycle library run with the ridge cycle model (ko_library/) vs the HistGB model (ko_library_hgb/).
Light: reads two 1,000-row CSVs. usage: python compare_ko_ridge_hgb.py [RIDGE_DIR] [HGB_DIR] [OUT_DIR]"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

RD = sys.argv[1] if len(sys.argv) > 1 else "ko_library"
HD = sys.argv[2] if len(sys.argv) > 2 else "ko_library_hgb"
OUT = sys.argv[3] if len(sys.argv) > 3 else "ko_library_hgb"
r = pd.read_csv(f"{RD}/ko_library_cycle.csv").set_index("gene"); h = pd.read_csv(f"{HD}/ko_library_cycle.csv").set_index("gene")
g = r.index.intersection(h.index); r, h = r.loc[g], h.loc[g]
print(f"genes in both: {len(g)}")
sr, sh = set(g[r.q_cycle < .05]), set(g[h.q_cycle < .05])
print(f"q_cycle<0.05: ridge {len(sr)}, HistGB {len(sh)}, both {len(sr & sh)}, ridge only {len(sr - sh)}, HistGB only {len(sh - sr)}")
print(f"Jaccard {len(sr & sh) / max(len(sr | sh), 1):.2f}")
for c in ["q_within", "q_4N"]:
    print(f"{c}<0.05: ridge {int((r[c] < .05).sum())}, HistGB {int((h[c] < .05).sum())}, both {int(((r[c] < .05) & (h[c] < .05)).sum())}")
print("class counts:\n" + pd.concat([r["class"].value_counts().rename("ridge"), h["class"].value_counts().rename("HistGB")], axis=1).fillna(0).astype(int).to_string())
print(f"Spearman across all KOs, ridge vs HistGB: cycle_TV {spearmanr(r.cycle_TV, h.cycle_TV)[0]:+.2f}, d_pct_4N_pred {spearmanr(r.d_pct_4N_pred, h.d_pct_4N_pred)[0]:+.2f}")
for nm, d in [("ridge", r), ("HistGB", h)]:
    ok = d.measured_pRb_shift_log2.notna()
    rho = spearmanr(d.d_pct_4N_pred[ok], d.measured_pRb_shift_log2[ok])[0]
    sig = ok & (d.q_cycle < .05)
    rho_s = spearmanr(d.d_pct_4N_pred[sig], d.measured_pRb_shift_log2[sig])[0] if sig.sum() > 3 else np.nan
    print(f"{nm}: predicted d%4N vs measured pRb shift, all {ok.sum()} KOs rho {rho:+.3f}; among {sig.sum()} q_cycle<0.05 KOs rho {rho_s:+.3f}")
    print(f"{nm}: heterogeneity median spread change {d.spread_change_pct.median():+.2f}% (shuffled {d.spread_change_shuffled_pct.median():+.3f}%), "
          f"median AUC change {d.auc_change.median():+.4f}, separate before {int(d.separates_before.sum())}, after {int(d.separates_after.sum())}, "
          f"median cycle_share_of_shift {d.cycle_share_of_shift.median():+.3f}")
cols = ["n", "cycle_TV", "q_cycle", "d_pct_4N_pred", "p_4N", "q_within", "cycle_share_of_shift", "auc_before", "auc_after", "measured_pRb_shift_log2"]
M = r[cols].add_suffix("_ridge").join(h[cols].drop(columns="n").add_suffix("_hgb"))
M["sig_ridge"] = M.q_cycle_ridge < .05; M["sig_hgb"] = M.q_cycle_hgb < .05
M.to_csv(f"{OUT}/compare_ridge_vs_hgb.csv")
show = ["n_ridge", "cycle_TV_ridge", "cycle_TV_hgb", "q_cycle_ridge", "q_cycle_hgb", "d_pct_4N_pred_ridge", "d_pct_4N_pred_hgb", "measured_pRb_shift_log2_ridge"]
print("\nfocus genes:\n" + M.reindex(["CDK7", "CCNB1", "POLA1", "CDC45", "MCM7"])[show].round(3).to_string())
print("\nbiggest HistGB-only additions (by HistGB cycle_TV):\n" + M[M.sig_hgb & ~M.sig_ridge].sort_values("cycle_TV_hgb", ascending=False).head(20)[show].round(3).to_string())
print("\nridge-only (lost with HistGB):\n" + M[M.sig_ridge & ~M.sig_hgb].sort_values("cycle_TV_ridge", ascending=False).head(20)[show].round(3).to_string())
print("\ntop 15 HistGB cycle changers:\n" + M[M.sig_hgb].sort_values("cycle_TV_hgb", ascending=False).head(15)[show].round(3).to_string())
print("DONE")
