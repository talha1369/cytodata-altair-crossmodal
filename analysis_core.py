"""Altair cross-modal inspection: phase-imaging morphology vs scRNA-seq gene embeddings."""
import anndata as ad, numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_decomposition import CCA
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

rna = ad.read_h5ad("data/cropseq_svaeplus_embedding_phate.h5ad")   # 1001 x 50, index=gene
img = ad.read_h5ad("data/ops_gene_embedding_phate_phase_only.h5ad")# 1052 x 97, obs.perturbation=gene

rna_emb_all = rna.obsm["X_sVAE"]
rna_idx = rna.obs.index.astype(str)
img_emb_all = np.asarray(img.X)
img_idx = img.obs["perturbation"].astype(str).values

CTRL = {"control","nontargeting","non-targeting","NTC","safe"}
# shared genes, excluding controls
shared = [g for g in rna_idx if g in set(img_idx) and g not in CTRL]
print(f"[match] shared non-control genes: {len(shared)}")

rna_pos = {g:i for i,g in enumerate(rna_idx)}
img_pos = {g:i for i,g in enumerate(img_idx)}
R = rna_emb_all[[rna_pos[g] for g in shared]]
I = img_emb_all[[img_pos[g] for g in shared]]
R = StandardScaler().fit_transform(R)
I = StandardScaler().fit_transform(I)
print(f"[shapes] RNA {R.shape}  IMG {I.shape}")

# ---- 1. Structure preservation (Mantel-style) ----
Sr = cosine_similarity(R); Si = cosine_similarity(I)
iu = np.triu_indices(len(shared), k=1)
vr, vi = Sr[iu], Si[iu]
rho, _ = spearmanr(vr, vi)
# permutation null: shuffle gene labels of IMG
nperm=500; null=np.empty(nperm)
for p in range(nperm):
    perm = rng.permutation(len(shared))
    Sip = Si[np.ix_(perm,perm)]
    null[p] = spearmanr(vr, Sip[iu]).statistic
z = (rho - null.mean())/null.std()
print(f"[structure] Spearman(gene-gene cos sim: RNA vs IMG) rho={rho:.3f} | null={null.mean():.3f}+/-{null.std():.3f} | z={z:.1f}")

# ---- 2. kNN neighborhood overlap vs null ----
def knn_sets(S,k):
    idx=np.argsort(-S,axis=1)[:,1:k+1]; return [set(row) for row in idx]
for k in (10,20):
    nr,ni = knn_sets(Sr,k), knn_sets(Si,k)
    ov = np.mean([len(nr[i]&ni[i])/k for i in range(len(shared))])
    nulls=[]
    for p in range(200):
        perm=rng.permutation(len(shared)); inv={perm[i]:i for i in range(len(shared))}
        nip=[set(inv[x] for x in ni[perm[i]]) for i in range(len(shared))]
        nulls.append(np.mean([len(nr[i]&nip[i])/k for i in range(len(shared))]))
    nulls=np.array(nulls)
    print(f"[kNN k={k}] cross-modal neighbor overlap={ov:.3f} | null={nulls.mean():.3f}+/-{nulls.std():.3f} | fold={ov/nulls.mean():.1f}x")

# ---- 3. CCA ----
nc=10
cca=CCA(n_components=nc, max_iter=1000); Rc,Ic=cca.fit_transform(R,I)
ccorr=[np.corrcoef(Rc[:,i],Ic[:,i])[0,1] for i in range(nc)]
print("[CCA] top canonical correlations:", np.round(ccorr,3))

# ---- 4. CORUM biological validation ----
col = "corum_complex" if "corum_complex" in img.obs else None
if col:
    cc = img.obs.set_index("perturbation")[col].astype(str)
    comp = np.array([cc.get(g,"") for g in shared])
    has = (comp!="") & (comp!="nan")
    print(f"[corum] genes with a complex: {has.sum()}")
    same=np.zeros_like(Sr,dtype=bool)
    for c in pd.unique(comp[has]):
        members=np.where(comp==c)[0]
        for a in members:
            for b in members:
                if a<b: same[a,b]=True
    smask=same[iu]
    for name,S in [("RNA",Sr),("IMG-phase",Si)]:
        v=S[iu]
        print(f"[corum] {name}: same-complex cos={v[smask].mean():.3f}  vs other={v[~smask].mean():.3f}  (n_same_pairs={smask.sum()})")
print("DONE")
