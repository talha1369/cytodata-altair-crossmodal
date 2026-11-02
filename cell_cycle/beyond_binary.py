"""Can morphology resolve cell-cycle stage BEYOND G1 vs G2/M? One-vs-rest held-out AUC for each 4i stage, and the
restriction-point split (early-G1 vs late-G1, same DNA, differ in pRb)."""
import os, numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def find(n):
    for r in ['/srv/team/data/crossmodal/ops0077_ntc', 'data', 'cell_cycle_4i_DNA', '.']:
        if os.path.exists(f'{r}/{n}'): return f'{r}/{n}'
    raise FileNotFoundError(n)


fr = pd.read_parquet(find('OPS0077_NTC_cellcycle_frame.parquet'))
rel = (fr['dna_reliable'].astype(str) == 'True').values
ph = ad.read_h5ad(find('OPS0077_NTC_phase_OrganelleProfiler.h5ad')); ph.obs_names = ph.obs['cell_key'].astype(str)
cells = fr.index[rel]; cells = cells[cells.isin(ph.obs_names)]
ph = ph[cells]; F = fr.loc[cells]
present = np.isfinite(np.asarray(ph.X)).mean(0) >= 0.85
Xm = np.asarray(ph.X, np.float32)[:, present]
groups = F['tile_pheno'].astype(str).values
stage = F['cycle_stage'].astype(str).values
print('stage counts:', pd.Series(stage).value_counts().to_dict())


def prep(a, b):
    imp = SimpleImputer(strategy='median').fit(a); sc = RobustScaler(quantile_range=(10, 90)).fit(imp.transform(a))
    f = lambda Q: np.clip(sc.transform(imp.transform(Q)), -5, 5)
    p = PCA(100, random_state=0).fit(f(a)); return p.transform(f(a)), p.transform(f(b))


def oof_auc_binary(X, y, g):
    pp = np.zeros(len(y), float)
    for a, b in GroupKFold(5).split(X, y, g):
        A, B = prep(X[a], X[b]); pp[b] = LogisticRegression(C=.1, max_iter=3000).fit(A, y[a]).predict_proba(B)[:, 1]
    return roc_auc_score(y, pp)


# one-vs-rest AUC per stage (can morphology pick out each stage, including S?)
print('\none-vs-rest held-out AUC from morphology:')
for st in ['G0/G1-early', 'G1-late', 'S', 'G2/M']:
    y = (stage == st).astype(int)
    print(f'  {st:12s} vs rest: {oof_auc_binary(Xm, y, groups):.3f}  (n={int(y.sum()):,})')

# the hard one: restriction point. early-G1 vs late-G1, SAME DNA (both 2N), differ in pRb only
m = np.isin(stage, ['G0/G1-early', 'G1-late'])
y = (stage[m] == 'G1-late').astype(int)
print(f'\nrestriction point  early-G1 vs late-G1 (both 2N, differ in pRb): AUC {oof_auc_binary(Xm[m], y, groups[m]):.3f} '
      f'(n early={int((y==0).sum()):,}, late={int((y==1).sum()):,})')

# S vs the rest of cycling cells (is S-phase morphology distinguishable?)
m2 = np.isin(stage, ['G1-late', 'S', 'G2/M'])
y2 = (stage[m2] == 'S').astype(int)
print(f'S phase vs (G1-late + G2/M): AUC {oof_auc_binary(Xm[m2], y2, groups[m2]):.3f} (n S={int((y2==1).sum()):,})')
