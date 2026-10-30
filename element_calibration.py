"""Cross-modal (RNA <-> imaging morphology) matching calibration on the Element true-pair screen.
Ground-truth reference for a future pooled screen. Axes (lasso_ev spec), shuffled null each:
 1) cells-per-group pooling k=1,10,50,100,200,400 within pooled condition blocks
 2) feature panel: full vs cell-paint stains vs a ~26-feat interpretable biology panel (shape+channel intensities+nuclear DNA)
 3) extent: whole-cell vs nuclear-region (_Nuclear suffix)
 4) link-feature degradation (noise) -> accuracy vs achieved imaging->RNA R2"""
import anndata as ad, numpy as np, pandas as pd, scanpy as sc, warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore"); rng=np.random.default_rng(0); alphas=np.logspace(-1,4,8)
rna=ad.read_h5ad("data/element_rna.h5ad"); img=ad.read_h5ad("data/element_imaging.h5ad")
ok=~rna.obs["qc_fail"].values; idx=np.where(ok)[0]
sub=rng.choice(idx,size=min(8000,len(idx)),replace=False); sub.sort()
R=rna[sub].copy(); I=img[sub].copy(); cond=R.obs["condition"].astype(str).values
sc.pp.normalize_total(R,target_sum=1e4); sc.pp.log1p(R); sc.pp.highly_variable_genes(R,n_top_genes=2000)
R=R[:,R.var.highly_variable].copy(); sc.pp.scale(R,max_value=10); sc.pp.pca(R,n_comps=50)
Yall=R.obsm["X_pca"]
names=I.var_names.astype(str).values; mod=I.var["modality"].astype(str).values; chan=I.var["channel"].astype(str).values
Xraw=np.nan_to_num(np.asarray(I.X,dtype=float))
real=np.isin(mod,["cell_paint","protein","shape"])
nuclear=np.array([n.endswith("_Nuclear") for n in names]) & real
whole=real & ~nuclear
# interpretable ~26-feat panel: shape feats + per cell-paint channel whole-cell mean + nuclear DNA(Nucleus) mean
Xs=StandardScaler().fit_transform(Xraw)
panel_cols=[Xs[:,(mod=="shape")]]                     # morphology/shape
cp_channels=[c for c in ["Mitochondria","Cell-Membrane","Nucleus","Actin"]]
for c in cp_channels:
    m=(chan==c)&whole
    if m.sum(): panel_cols.append(Xs[:,m].mean(1,keepdims=True))   # channel whole-cell intensity
mn=(chan=="Nucleus")&nuclear
if mn.sum(): panel_cols.append(Xs[:,mn].mean(1,keepdims=True))     # nuclear DNA
PANEL=np.hstack(panel_cols)
def oof(X,Y):
    O=np.zeros_like(Y)
    for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
        xs,ys=StandardScaler().fit(X[tr]),StandardScaler().fit(Y[tr])
        O[te]=ys.inverse_transform(RidgeCV(alphas=alphas).fit(xs.transform(X[tr]),ys.transform(Y[tr])).predict(xs.transform(X[te])))
    return O
def retr(P,A):
    S=cosine_similarity(P,A); rk=np.array([np.where(np.argsort(-S[i])==i)[0][0] for i in range(len(P))])
    return float(np.mean(rk<10)), float(np.median(rk)+1)
rows=[]
Xreal=Xs[:,real]
# AXIS 1 pooling
for k in [1,10,50,100,200,400]:
    Xg=[];Yg=[]
    for c in np.unique(cond):
        ci=np.where(cond==c)[0].copy(); rng.shuffle(ci); ng=len(ci)//k
        for g in range(ng): gi=ci[g*k:(g+1)*k]; Xg.append(Xreal[gi].mean(0)); Yg.append(Yall[gi].mean(0))
    Xg=np.array(Xg);Yg=np.array(Yg);P=oof(Xg,Yg);t10,med=retr(P,Yg);Pn=P[rng.permutation(len(P))];t10n,_=retr(Pn,Yg)
    rows.append(dict(axis="cells_per_group",k=k,n_groups=len(Xg),r2=r2_score(Yg,P,multioutput="uniform_average"),top10=t10,med_pct=med/len(Xg),null_top10=t10n))
    print("A1",rows[-1],flush=True)
# AXIS 2 panel (k=1)
for name,M in [("full",Xreal),("cell_paint_stains",Xs[:,(mod=="cell_paint")]),("interpretable_panel",PANEL)]:
    P=oof(M,Yall);t10,med=retr(P,Yall);Pn=P[rng.permutation(len(P))];t10n,_=retr(Pn,Yall)
    rows.append(dict(axis="panel",which=name,n_feat=M.shape[1],r2=r2_score(Yall,P,multioutput="uniform_average"),top10=t10,null_top10=t10n))
    print("A2",rows[-1],flush=True)
# AXIS 3 extent (k=1)
for name,M in [("whole_cell",Xs[:,whole]),("nuclear_region",Xs[:,nuclear])]:
    if M.shape[1]<3: continue
    P=oof(M,Yall);t10,med=retr(P,Yall)
    rows.append(dict(axis="extent",which=name,n_feat=M.shape[1],r2=r2_score(Yall,P,multioutput="uniform_average"),top10=t10))
    print("A3",rows[-1],flush=True)
# AXIS 4 link strength (k=1)
bs=Xreal.std()
for lvl in [0,0.5,1,2,4]:
    Xn=Xreal+rng.normal(0,lvl*bs,Xreal.shape);P=oof(Xn,Yall);t10,_=retr(P,Yall)
    rows.append(dict(axis="link_strength",noise=lvl,r2=r2_score(Yall,P,multioutput="uniform_average"),top10=t10))
    print("A4",rows[-1],flush=True)
df=pd.DataFrame(rows); df.to_csv("results/element_matching_calibration.csv",index=False)
fig,ax=plt.subplots(1,4,figsize=(19,4.1))
a1=df[df.axis=="cells_per_group"];ax[0].plot(a1.k,a1.top10*100,"o-",label="observed");ax[0].plot(a1.k,a1.null_top10*100,"s--",color="0.6",label="null");ax[0].set_xscale("log");ax[0].set_xlabel("cells/group");ax[0].set_ylabel("top-10 retrieval %");ax[0].set_title("Pooling");ax[0].legend(frameon=False)
a2=df[df.axis=="panel"];ax[1].bar(a2.which,a2.r2,color=["#4C78A8","#E49444","#59A14F"]);ax[1].set_ylabel("held-out R2");ax[1].set_title("Feature panel");ax[1].tick_params(axis="x",rotation=20)
a3=df[df.axis=="extent"];ax[2].bar(a3.which,a3.r2,color=["#4C78A8","#B07AA1"]);ax[2].set_ylabel("held-out R2");ax[2].set_title("Extent: whole vs nuclear")
a4=df[df.axis=="link_strength"];ax[3].plot(a4.r2,a4.top10*100,"o-");ax[3].set_xlabel("imaging->RNA R2 (degraded)");ax[3].set_ylabel("top-10 retrieval %");ax[3].set_title("Matching vs link strength")
plt.tight_layout();plt.savefig("figures/element/matching_calibration.png",bbox_inches="tight",dpi=150)
print("DONE saved results/element_matching_calibration.csv + figures/element/matching_calibration.png",flush=True)
