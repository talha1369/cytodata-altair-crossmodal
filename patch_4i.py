import nbformat as nbf
nb=nbf.read("guide_level_cross_modal.ipynb",as_version=4)

md_4i="""## 5. What the fixed 4i panel adds (Brian's OPS0077 idea)
OPS0077 is the 4i *fixed* screen — the same cells carry γH2AX, p21, p53, pRb, Rb and RPS6 (DNA-damage / cell-cycle / translation stress). Does this panel add cross-modal signal over label-free phase for predicting the transcriptome? We join each marker's guide profile and reduce **each modality to 40 PCs** so the comparison is size-balanced (otherwise the 1,655 marker features swamp the 337 phase features)."""

code_4i=r'''markers4i=["gh2ax","p21","p53","prb","rb","rsp6"]
def load_sel(fn):
    a=ad.read_h5ad(DATA/fn); a=a[:,a.var["selected"].values].copy()
    a.X=np.clip(np.nan_to_num(a.X),-50,50); return a
mk={m:load_sel(f"{m}_guide.h5ad") for m in markers4i}
common=set(ph.obs["guide"].astype(str))
for m in markers4i: common&=set(mk[m].obs["guide"].astype(str))
pgx=ph.obs.set_index(ph.obs["guide"].astype(str))
gid=sorted(x for x in common if x in emb and pgx.loc[x,"target_gene"]!="NTC")
pix={x:i for i,x in enumerate(ph.obs["guide"].astype(str))}
mixm={m:{x:i for i,x in enumerate(mk[m].obs["guide"].astype(str))} for m in markers4i}
PHc=np.vstack([ph.X[pix[x]] for x in gid])
MKc=np.hstack([np.vstack([mk[m].X[mixm[m][x]] for x in gid]) for m in markers4i])
Rc=np.vstack([emb[x] for x in gid]); nc=len(gid)
from sklearn.decomposition import PCA
def oof_pca(blocks,Y,Kp=40):
    O=np.zeros_like(Y)
    for tr,te in KFold(5,shuffle=True,random_state=0).split(Y):
        Xt=[];Xe=[]
        for B in blocks:
            p=PCA(n_components=min(Kp,B.shape[1]),random_state=0).fit(B[tr])
            Xt.append(p.transform(B[tr])); Xe.append(p.transform(B[te]))
        ys=StandardScaler().fit(Y[tr])
        O[te]=ys.inverse_transform(RidgeCV(alphas=alphas).fit(np.hstack(Xt),ys.transform(Y[tr])).predict(np.hstack(Xe)))
    return O
r4={}
for nm,bl in [("phase",[PHc]),("4i",[MKc]),("phase+4i",[PHc,MKc])]:
    O=oof_pca(bl,Rc); r4[nm]=(r2_score(Rc,O,multioutput="uniform_average"), retr(O,Rc)["top10"])
    print(f"[{nm:9s}->RNA]  R2={r4[nm][0]:.3f}  top10={r4[nm][1]:.3f}  (n={nc})")
fig,ax=plt.subplots(figsize=(5.6,4)); nmz=list(r4); xx=np.arange(len(nmz))
ax.bar(xx,[r4[k][0] for k in nmz],color=["#4C78A8","#E49444","#59A14F"],width=.6)
for i,k in enumerate(nmz): ax.text(i,r4[k][0]+.008,f"{r4[k][0]:.2f}",ha="center")
ax.set_xticks(xx); ax.set_xticklabels(nmz); ax.set_ylim(0,0.45)
ax.set_ylabel("held-out R2 (-> RNA)"); ax.set_title("Phase vs the 4i stress panel for predicting RNA (40 PCs/modality)")
plt.tight_layout(); plt.savefig(FIG/"phase_vs_4i.png",bbox_inches="tight",dpi=150); plt.show()'''

takeaways="""## 6. Takeaways
- Morphology↔expression is predictable at single-guide resolution — phase→RNA R²≈0.34, RNA→phase R²≈0.61; cross-validated CCA keeps ~5 shared axes at r≈0.8–0.95.
- Aggregating guides→genes sharpens the link — the cross-modal signal concentrates at the gene level (single guides are noisy), echoing the known result that single-cell multimodal MoA is hard.
- **Phase ≫ the 4i stress panel for predicting the transcriptome** (R²≈0.33 vs ≈0.10), and **phase+4i ≈ phase** — this fixed γH2AX / cell-cycle / translation panel is largely redundant with label-free phase for cross-modal prediction. (Panel-specific: a broader marker set could add more.)
- Where one modality can't predict another marks modality-specific biology — the argument for keeping label-free imaging alongside transcriptomics.

**Caveats:** CROP-seq cells ≠ imaged cells (cross-modal only at perturbation level); mantispy-selected OrganelleProfiler features, not a learned/Cell-DINO embedding; linear predictors are a floor; the 4i result is specific to this stress-marker panel."""

# 1) add 4i download to the data cell
for c in nb.cells:
    if c.cell_type=="code" and "cropseq_svaeplus.h5ad" in c.source and "phase_guide.h5ad" in c.source:
        c.source=c.source.replace('print("data ready")',
'''MARKERS4I=["gh2ax","p21","p53","prb","rb","rsp6"]
if not all((DATA/f"{m}_guide.h5ad").exists() for m in MARKERS4I):
    os.system('curl -s -o data/markers_guide.zip https://exampledata.scverse.org/sbi2-cytodata/ops/markers_guide.zip')
    with zipfile.ZipFile(DATA/"markers_guide.zip") as zz:
        for m in MARKERS4I: zz.extract(f"{m}_guide.h5ad", DATA)
print("data ready")''')
        break

# 2) find takeaways cell, insert 4i md+code before it, renumber takeaways
ti=[i for i,c in enumerate(nb.cells) if c.cell_type=="markdown" and "Takeaways" in c.source][0]
nb.cells[ti].source=takeaways
nb.cells[ti:ti]=[nbf.v4.new_markdown_cell(md_4i), nbf.v4.new_code_cell(code_4i)]
nbf.write(nb,"guide_level_cross_modal.ipynb")
print("patched; cells now",len(nb.cells))
