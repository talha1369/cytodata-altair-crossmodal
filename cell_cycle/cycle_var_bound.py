import numpy as np, pandas as pd, anndata as ad
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
L=pd.read_csv("cell_cycle_scores.csv").set_index("cell_key"); L=L[(L.dna_reliable==True)&(L.cc_phase_source=="dna+pcs")]
ph=ad.read_h5ad("OPS0077_NTC_phase_OrganelleProfiler.h5ad"); ph.obs_names=ph.obs["cell_key"].astype(str); ph=ph[ph.obs_names.isin(L.index)]; L=L.loc[ph.obs_names]
X=np.asarray(ph.X,np.float32); X=X[:, np.isfinite(X).mean(0)>=.85]
imp=SimpleImputer(strategy="median").fit(X); sc=RobustScaler(quantile_range=(10,90)).fit(imp.transform(X)); Z0=np.clip(sc.transform(imp.transform(X)),-5,5)
Z=PCA(100,random_state=0).fit_transform(Z0)
def harm(p,k=2):
    t=2*np.pi*p; return np.column_stack([f(j*t) for j in range(1,k+1) for f in (np.sin,np.cos)])
def removed(Z,C):
    C=np.c_[C,np.ones(len(C))]; R=Z-C@np.linalg.lstsq(C,Z,rcond=None)[0]; return 100*(1-np.trace(np.cov(R.T))/np.trace(np.cov((Z-Z.mean(0)).T)))
dna=L.dna_rel_R234.astype(float).values; zz=L[["z_dna","z_pRb","z_p21","z_phos"]].astype(float).values
print("percent of NTC phase-morphology variance (PCA-100) explained by:")
for nm,C in [("measured cc_phase, harmonics 1-2",harm(L.cc_phase.astype(float).values)),("measured cc_phase, harmonics 1-4",harm(L.cc_phase.astype(float).values,4)),
             ("measured DNA content (cubic)",np.c_[dna,dna**2,dna**3]),("measured DNA+pRb+p21+phos (linear+squares)",np.c_[zz,zz**2]),
             ("shuffled cc_phase",harm(np.random.default_rng(0).permutation(L.cc_phase.astype(float).values)))]:
    print(f"  {nm:46s} {removed(Z,C):5.1f}%")
pv=PCA(100,random_state=0).fit(Z0).explained_variance_ratio_; print("for scale: PC1 alone explains",round(100*pv[0]/pv.sum(),1),"% of the PCA-100 variance")
