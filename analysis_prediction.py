"""Held-out cross-modal prediction: phase-imaging morphology <-> scRNA-seq gene embeddings.
Out-of-fold (5-fold, held-out GENES), per-fold scaling (no leakage), RidgeCV.
Metrics: OOF R^2, cross-modal retrieval (top-k, median rank), cross-validated CCA, per-gene quality."""
import anndata as ad, numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_decomposition import CCA
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

rna = ad.read_h5ad("data/cropseq_svaeplus_embedding_phate.h5ad")
img = ad.read_h5ad("data/ops_gene_embedding_phate_phase_only.h5ad")
rna_idx = rna.obs.index.astype(str); img_idx = img.obs["perturbation"].astype(str).values
CTRL = {"control","nontargeting","non-targeting","NTC","safe"}
shared = [g for g in rna_idx if g in set(img_idx) and g not in CTRL]
rp={g:i for i,g in enumerate(rna_idx)}; ip={g:i for i,g in enumerate(img_idx)}
RNA = rna.obsm["X_sVAE"][[rp[g] for g in shared]].astype(float)
IMG = np.asarray(img.X)[[ip[g] for g in shared]].astype(float)
n = len(shared); print(f"shared genes: {n} | RNA {RNA.shape} | IMG(phase) {IMG.shape}")
alphas = np.logspace(-1,4,12)

def oof_predict(X, Y):
    """out-of-fold multi-output RidgeCV preds for Y from X; per-fold scaling."""
    P = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        xs, ys = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
        m = RidgeCV(alphas=alphas).fit(xs.transform(X[tr]), ys.transform(Y[tr]))
        P[te] = ys.inverse_transform(m.predict(xs.transform(X[te])))
    return P

def retrieval(pred, actual):
    """rank of each gene's true match when querying actual gallery with its prediction."""
    S = cosine_similarity(pred, actual)
    order = np.argsort(-S, axis=1)
    ranks = np.array([np.where(order[i]==i)[0][0] for i in range(len(pred))])  # 0-based
    return {"top1":np.mean(ranks==0),"top5":np.mean(ranks<5),"top10":np.mean(ranks<10),
            "median_rank":np.median(ranks)+1}, ranks

for name, X, Y in [("morphology(phase) -> RNA", IMG, RNA), ("RNA -> morphology(phase)", RNA, IMG)]:
    P = oof_predict(X, Y)
    r2u = r2_score(Y, P, multioutput="uniform_average")
    r2v = r2_score(Y, P, multioutput="variance_weighted")
    met,_ = retrieval(P, Y)
    # null retrieval: shuffle pairing
    Pn = P[rng.permutation(n)]; metn,_ = retrieval(Pn, Y)
    print(f"\n[{name}]")
    print(f"  held-out R^2: uniform={r2u:.3f} | variance-weighted={r2v:.3f}")
    print(f"  retrieval (gallery={n}): top1={met['top1']:.3f} top5={met['top5']:.3f} top10={met['top10']:.3f} median_rank={met['median_rank']:.0f}")
    print(f"  chance:    top1={1/n:.4f} top5={5/n:.4f} top10={10/n:.4f} median_rank~{n/2:.0f}  (null top1={metn['top1']:.4f})")

# cross-validated CCA (honest version of the earlier in-sample numbers)
nc=10; test_cc=np.zeros((5,nc))
for f,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=0).split(RNA)):
    xs,ys=StandardScaler().fit(RNA[tr]),StandardScaler().fit(IMG[tr])
    cca=CCA(n_components=nc,max_iter=2000).fit(xs.transform(RNA[tr]),ys.transform(IMG[tr]))
    a,b=cca.transform(xs.transform(RNA[te]),ys.transform(IMG[te]))
    test_cc[f]=[np.corrcoef(a[:,i],b[:,i])[0,1] for i in range(nc)]
print("\n[CCA held-out] mean test canonical correlations:", np.round(test_cc.mean(0),3))

# per-gene quality (morphology->RNA): cosine(pred, actual) per gene, held-out
P = oof_predict(IMG, RNA)
pg = np.array([np.corrcoef(P[i],RNA[i])[0,1] for i in range(n)])
dfq = pd.DataFrame({"gene":shared,"predicted_vs_actual_r":pg})
sc = img.obs.set_index("perturbation")["supercategory"].reindex(shared).astype(str).values
dfq["supercategory"]=sc
print("\n[per-gene morphology->RNA predictability]")
print("  best:", list(dfq.sort_values('predicted_vs_actual_r',ascending=False).head(8)['gene']))
print("  worst:", list(dfq.sort_values('predicted_vs_actual_r').head(8)['gene']))
top=dfq.nlargest(150,'predicted_vs_actual_r')['supercategory'].value_counts(normalize=True).head(4)
bot=dfq.nsmallest(150,'predicted_vs_actual_r')['supercategory'].value_counts(normalize=True).head(4)
print("  categories enriched among BEST-predicted:", dict(top.round(2)))
print("  categories enriched among WORST-predicted:", dict(bot.round(2)))
dfq.to_csv("per_gene_predictability.csv", index=False)
print("\nwrote per_gene_predictability.csv ; DONE")
