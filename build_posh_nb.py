import nbformat as nbf
nb=nbf.v4.new_notebook(); cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s)); code=lambda s: cells.append(nbf.v4.new_code_cell(s))
md("""# Altair vs CellPainting-POSH: within-cell-line cross-screen reproducibility
Two independent optical/Cell-Painting screens, **both in A549**:
- **Altair** (phase morphology, gene-level from Tim Treis's prep-ops `phase_guide.h5ad`).
- **CellPainting-POSH** (A549; added to the shared repo).

Because both are A549, this is a **within-cell-line** comparison, which removes the cell-line confounder that weakened the Altair-vs-Aconcagua (A549-vs-HeLa) comparison. Question: do two independent A549 imaging screens recover the same gene-function structure, and which gene relationships reproduce. Same feature spaces are not identical across screens, so we compare relationship structure (gene-gene similarity), not raw features.

This notebook is **manifest-driven and runs the moment POSH is present**: it finds POSH via `download_manifest.json` or a `data/*posh*` file, auto-detects the gene column and feature matrix, and otherwise prints a clear "awaiting POSH" and skips. The cross-screen machinery is the same as `altair_vs_aconcagua.ipynb`.""")
code("""import os, json, glob, warnings
import anndata as ad, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); rng=np.random.default_rng(0)
FIG="figures/cross_screen"; os.makedirs(FIG,exist_ok=True)

def find_posh():
    if os.path.exists("download_manifest.json"):
        man=json.load(open("download_manifest.json"))
        for f in man.get("files",[]):
            tag=(str(f.get("dataset",""))+" "+str(f.get("name",""))+" "+str(f.get("path",""))).lower()
            if "posh" in tag and str(f.get("path","")).endswith(".h5ad"):
                for cand in (os.path.join(man.get("data_dir","data"), f["path"]), os.path.join("data", f["path"])):
                    if os.path.exists(cand): return cand
    for pat in ["data/**/*[pP][oO][sS][hH]*.h5ad","data/*[pP][oO][sS][hH]*.h5ad"]:
        g=sorted(glob.glob(pat,recursive=True))
        if g: return g[0]
    return None

GENE_COLS=["gene_symbol","target_gene","gene","perturbation_id","perturbation","gene_id","geneKO_name"]
def load_gene_profile(path, name):
    a=ad.read_h5ad(path)
    # gene labels
    gcol=next((c for c in GENE_COLS if c in a.obs.columns), None)
    genes=(a.obs[gcol].astype(str).values if gcol else a.obs_names.astype(str).values)
    # feature matrix: prefer a dense embedding in obsm, else X if 2D-usable
    if "X_pca" in a.obsm: Xf=np.asarray(a.obsm["X_pca"])
    elif "X_sVAE" in a.obsm: Xf=np.asarray(a.obsm["X_sVAE"])
    else:
        X=a.X; Xf=np.asarray(X.todense()) if hasattr(X,"todense") else np.asarray(X)
    if "selected" in a.var and Xf.shape[1]==a.n_vars and a.var["selected"].sum()>5:
        Xf=Xf[:, a.var["selected"].values]
    Xf=np.clip(np.nan_to_num(Xf.astype(float)),-50,50)
    df=pd.DataFrame(Xf, index=pd.Index(genes))
    df=df[~df.index.isin(["NTC","control","nontargeting",""])]
    g=df.groupby(level=0).mean()
    print(f"{name}: {path} -> {g.shape[0]} genes x {g.shape[1]} feat (gene col={gcol})")
    return g

posh_path=find_posh()
# Altair A549 phase, gene level (always available locally)
ph=ad.read_h5ad("data/phase_guide.h5ad"); ph=ph[:,ph.var["selected"].values].copy(); ph.X=np.clip(np.nan_to_num(ph.X),-50,50)
alt=pd.DataFrame(ph.X,index=pd.Index(ph.obs["target_gene"].astype(str))); alt=alt[alt.index!="NTC"].groupby(level=0).mean()
print(f"Altair phase: {alt.shape[0]} genes x {alt.shape[1]} feat")
if posh_path is None:
    posh=None; print("\\nAWAITING POSH: no POSH file found via download_manifest.json or data/*posh*.h5ad. Add it and rerun.")
else:
    posh=load_gene_profile(posh_path,"CellPainting-POSH (A549)")""")
md("""## Cross-screen gene-gene structure (runs when POSH is present)
Within each screen, z-score features and compute every gene-gene cosine similarity, then correlate the two similarity matrices over shared genes (Spearman, Mantel style) against a shuffled null, with 10-NN overlap. Expect stronger agreement than the A549-vs-HeLa case because the cell line is matched.""")
code("""if posh is None:
    print("Skipped: POSH not available yet.")
else:
    shared=sorted(set(alt.index)&set(posh.index)); print("shared A549 genes:",len(shared))
    A=StandardScaler().fit_transform(alt.loc[shared].values); P=StandardScaler().fit_transform(posh.loc[shared].values)
    Sa,Sp=cosine_similarity(A),cosine_similarity(P); iu=np.triu_indices(len(shared),1)
    rho=spearmanr(Sa[iu],Sp[iu]).statistic
    null=np.array([spearmanr(Sa[iu],Sp[np.ix_(q,q)][iu]).statistic for q in (rng.permutation(len(shared)) for _ in range(200))])
    z=(rho-null.mean())/null.std()
    def knn(S,k=10): return [set(np.argsort(-S[i])[1:k+1]) for i in range(len(S))]
    na,nph=knn(Sa),knn(Sp); ov=np.mean([len(na[i]&nph[i])/10 for i in range(len(shared))])
    print(f"Spearman rho(Altair phase vs POSH, both A549) = {rho:.3f} | null {null.mean():+.3f}+/-{null.std():.3f} | z={z:.0f}")
    print(f"10-NN overlap = {ov:.3f} (chance {10/len(shared):.4f}, {ov/(10/len(shared)):.1f}x)")
    fig,ax=plt.subplots(figsize=(5.4,5)); hb=ax.hexbin(Sa[iu],Sp[iu],gridsize=55,bins="log",cmap="viridis",mincnt=1)
    ax.set_xlabel("gene-gene cosine sim: Altair (phase)"); ax.set_ylabel("gene-gene cosine sim: POSH"); ax.set_title(f"Within-A549 cross-screen (n={len(shared)})\\nSpearman rho={rho:.2f}, z={z:.0f}")
    fig.colorbar(hb,label="log10 pair count"); plt.tight_layout(); plt.savefig(f"{FIG}/altair_vs_posh.png",bbox_inches="tight",dpi=150); plt.show()""")
md("""## Takeaways
- Both screens are A549, so a positive, strong gene-gene structure correlation here is **within-cell-line reproducibility** of imaging-based gene function, the clean version of the cross-screen question.
- Compare the rho here to the A549-vs-HeLa Altair-vs-Aconcagua value: the gap estimates how much of the earlier modest signal was cell-line difference vs screen/modality difference.
- Caveats: different feature spaces (relationship-based comparison), POSH is smaller and lightly QC'd, gene-level only.""")
nb["cells"]=cells; nb["metadata"]["kernelspec"]={"name":"altair","display_name":"altair","language":"python"}
nbf.write(nb,"altair_vs_posh.ipynb"); print("wrote altair_vs_posh.ipynb with",len(cells),"cells")
