import nbformat as nbf
nb=nbf.read("cross_modal_prediction.ipynb",as_version=4)
md_dino="""## 4. Cell-DINO vs OrganelleProfiler: which imaging representation predicts RNA best?
The Altair atlas ships two imaging representations: the hand-crafted **OrganelleProfiler phase** features used above, and a **learned Cell-DINO** embedding (the gene-level consensus in `atlas/aggregated_data.h5ad`, 66 sparse-PCs of the live-cell DINO model). We compare both, and their combination, for held-out prediction of the CROP-seq transcriptome, on the genes present in all three."""
code_dino=r'''import os
if not os.path.exists("data/aggregated_data.h5ad"):
    os.system('curl -s -o data/aggregated_data.h5ad "https://ops-explorer-public.s3.us-west-2.amazonaws.com/leonetti_ops/ops_data_portal_submission/v2.0.20260724/atlas/aggregated_data.h5ad"')
dino=ad.read_h5ad("data/aggregated_data.h5ad")
gcol=[c for c in ("geneKO_name","perturbation_id","gene_id") if c in dino.obs][0]
dmap={g:np.asarray(dino.X)[i] for i,g in enumerate(dino.obs[gcol].astype(str))}
pos={g:i for i,g in enumerate(shared)}
sh3=[g for g in shared if g in dmap]
R3=RNA[[pos[g] for g in sh3]]; I3=IMG[[pos[g] for g in sh3]]; D3=np.vstack([dmap[g] for g in sh3]); n3=len(sh3)
print(f"genes in Cell-DINO, phase and RNA: {n3} | Cell-DINO {D3.shape} phase {I3.shape} RNA {R3.shape}")
cmp={}
for name,X in [("Cell-DINO",D3),("phase (OrgProf)",I3),("Cell-DINO+phase",np.hstack([D3,I3]))]:
    P=oof_predict(X,R3); met,_=retrieval(P,R3)
    cmp[name]=dict(r2u=r2_score(R3,P,multioutput="uniform_average"),top10=met["top10"],med=met["median_rank"])
    print(f"[{name:16s}->RNA] R2={cmp[name]['r2u']:.3f} top10={cmp[name]['top10']:.3f} med_rank={cmp[name]['med']:.0f}/{n3}")
fig,ax=plt.subplots(1,2,figsize=(11,4)); nm=list(cmp); xx=np.arange(len(nm)); cols=["#4C78A8","#E49444","#59A14F"]
ax[0].bar(xx,[cmp[k]['r2u'] for k in nm],color=cols,width=.6)
for i,k in enumerate(nm): ax[0].text(i,cmp[k]['r2u']+.008,f"{cmp[k]['r2u']:.2f}",ha="center")
ax[0].set_xticks(xx); ax[0].set_xticklabels(nm,rotation=10); ax[0].set_ylabel("held-out R2 (-> RNA)"); ax[0].set_ylim(0,0.65)
ax[0].set_title("Which imaging representation predicts RNA best")
ax[1].bar(xx,[cmp[k]['top10']*100 for k in nm],color=cols,width=.6)
for i,k in enumerate(nm): ax[1].text(i,cmp[k]['top10']*100+.6,f"{cmp[k]['top10']*100:.0f}%",ha="center")
ax[1].axhline(10/n3*100,ls="--",c="k",lw=1); ax[1].set_xticks(xx); ax[1].set_xticklabels(nm,rotation=10)
ax[1].set_ylabel(f"top-10 retrieval % (gallery={n3})"); ax[1].set_title("Cross-modal retrieval (dashed = chance)")
plt.tight_layout(); plt.savefig("figures/prediction/cell_dino_vs_phase.png",bbox_inches="tight",dpi=150); plt.show()'''
ti=[i for i,c in enumerate(nb.cells) if c.cell_type=="markdown" and "Takeaways" in c.source][0]
old=nb.cells[ti].source
nb.cells[ti].source=old.replace("## 4. Takeaways","## 5. Takeaways").replace(
 "**Morphology ⟺ expression is strongly, bidirectionally predictable at the gene level**",
 "**The learned Cell-DINO embedding beats hand-crafted OrganelleProfiler phase** for predicting RNA (held-out R² about 0.50 vs 0.41, top-10 retrieval about 47% vs 37%); combining them adds a little. Use Cell-DINO for cross-modal work.\n- **Morphology to expression is strongly, bidirectionally predictable at the gene level**")
nb.cells[ti:ti]=[nbf.v4.new_markdown_cell(md_dino), nbf.v4.new_code_cell(code_dino)]
nbf.write(nb,"cross_modal_prediction.ipynb"); print("patched; cells now",len(nb.cells))
