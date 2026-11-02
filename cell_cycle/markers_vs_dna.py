#!/usr/bin/env python
"""How do the 4i markers-only and 4i DNA-based cell-cycle angles relate (same cells)?

Checks (a) whether the stored `cc_phase_markers` is rotationally aligned to `cc_phase_dna` (the alignment step in
ops0077_cell_cycle_score.py uses np.abs(mean(exp(i*diff))), which is rotation-INVARIANT, so its shift search cannot pick a
shift) and (b) the actual relation: markers angle as a function of DNA angle.   usage: python markers_vs_dna.py [OUT_DIR]"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "cc_align"
os.makedirs(OUT, exist_ok=True)


def find(n):
    for r in ["/srv/team/data/crossmodal/ops0077_ntc", "data", "cell_cycle_4i_DNA", "."]:
        if os.path.exists(f"{r}/{n}"): return f"{r}/{n}"
    raise FileNotFoundError(n)


d = pd.read_csv(find("cell_cycle_scores.csv")); d = d[d.cc_phase_markers.notna() & d.cc_phase_dna.notna()]
a, b = d.cc_phase_dna.values, d.cc_phase_markers.values
cd = lambda x, y: (x - y + .5) % 1 - .5
hers = lambda x, y: float(np.abs(np.mean(np.exp(2j * np.pi * cd(x, y)))))
print(f"{len(d):,} cells with both regimes")
print("her circ_corr with extra shifts added to the markers angle:", [round(hers((b + s) % 1, a), 6) for s in (0, .1, .25, .5, .75)])
z = np.mean(np.exp(2j * np.pi * (b - a)))
print(f"stored markers vs DNA: resultant length {abs(z):.3f}, mean phase difference {np.degrees(np.angle(z)) % 360:.0f} deg (0 would mean aligned)")

# rotation-SENSITIVE alignment: maximise Re(mean exp(i(f*b + shift - a)))
best = max(((np.mean(np.cos(2 * np.pi * ((f * b + sh - a)))), f, sh) for f in (1, -1) for sh in np.linspace(0, 1, 360, endpoint=False)))
sc, f, sh = best
print(f"rotation-sensitive alignment: flip {f:+d}, shift {sh * 360:.0f} deg, mean cos(difference) {sc:.3f}")
al = (f * b + sh) % 1
err = np.abs(cd(al, a)) * 360
print(f"after that alignment: median angular difference {np.median(err):.0f} deg, within 45 deg {np.mean(err < 45):.2f}  (random: 90 deg)")

bins = np.floor(a * 12).astype(int); rows = []
for k in range(12):
    m = bins == k; zz = np.mean(np.exp(2j * np.pi * b[m]))
    rows.append(dict(dna_angle_deg=f"{k * 30}-{k * 30 + 30}", n=int(m.sum()), markers_mean_angle_deg=round(float(np.degrees(np.angle(zz)) % 360)),
                     concentration=round(float(abs(zz)), 2)))
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/markers_vs_dna_relation.csv", index=False); print(tab.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a_, y_, ttl in [(ax[0], b * 360, "as stored"), (ax[1], al * 360, f"after rotation-sensitive alignment (flip {f:+d}, shift {sh * 360:.0f} deg)")]:
    h = a_.hexbin(a * 360, y_, gridsize=60, bins="log", extent=(0, 360, 0, 360), cmap="viridis")
    a_.set_xlabel("4i DNA-based clock angle (deg)"); a_.set_ylabel("4i markers-only clock angle (deg)"); a_.set_title(ttl)
    a_.plot([0, 360], [0, 360], "w--", lw=.8)
plt.colorbar(h, ax=ax, label="cells (log)", fraction=.025)
plt.savefig(f"{OUT}/cc_align_7_markers_vs_dna.png", dpi=110, bbox_inches="tight")
pd.DataFrame([dict(stored_resultant_length=abs(z), stored_mean_offset_deg=float(np.degrees(np.angle(z)) % 360),
                   best_flip=f, best_shift_deg=sh * 360, aligned_mean_cos=sc, aligned_median_diff_deg=float(np.median(err)))]).to_csv(f"{OUT}/markers_vs_dna_summary.csv", index=False)
print("DONE")
