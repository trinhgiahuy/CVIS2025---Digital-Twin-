"""Paper-style diagnostics; ratios/feature hooks are reconstruction, not exact paper code."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .processing import cdr_floor,image_rgb,static_to_total
from .records import read_array

def save_confusion(cm,path,title='Confusion matrix'):
    m=np.asarray(cm,dtype=int)
    fig,ax=plt.subplots(figsize=(5.4,4.4));im=ax.imshow(m,cmap='Blues')
    labs=['empty','1 person','2 people'][:len(m)]
    ax.set_xticks(range(len(m)),labs);ax.set_yticks(range(len(m)),labs)
    ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(title)
    for (i,j),v in np.ndenumerate(m):
        ax.text(j,i,str(v),ha='center',va='center',color='white' if v > max(1,m.max())/2 else 'black')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)

def plot_diagnostics(manifest,bounds,output,seed=42):
    from .records import read_manifest
    from .processing import calibration_stats
    df=read_manifest(manifest)
    def part(domain,split,label=None):
        x=df[(df.domain==domain)&(df.split==split)]
        if label is not None:x=x[x.label==label]
        return [read_array(r.path,r.kind) for r in x.itertuples(index=False)]
    sim=part('sim','train');real=part('real','test');cal=part('real','calibration')
    stats=calibration_stats(cal);rng=np.random.default_rng(seed)
    cdr=[cdr_floor(s,rng,stats['mu_db'],stats['sigma_db']) for s in sim]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Approximate distributions (subsample by pixel count to reduce memory)
    fig,ax=plt.subplots(figsize=(8,4))
    for label,arrs in [('Simulated RT',sim),('Real test (diagnostic ONLY)',real),('Simulated + CDR',cdr)]:
        vec=np.concatenate([x.ravel()[::20] for x in arrs])
        ax.hist(vec,bins=100,density=True,alpha=.4,label=label)
    ax.set_yscale('log');ax.set_xlabel('RD magnitude (dB)');ax.legend();fig.tight_layout()
    fig.savefig(out/'fig4_histogram_reconstructed.png',dpi=200);plt.close(fig)
    # Static-to-total scatter
    fig,ax=plt.subplots(figsize=(7,5))
    for label,arrs in [('RT',sim),('CDR',cdr),('Real',real)]:
        vals=np.asarray([static_to_total(s) for s in arrs]);ax.scatter(vals[:,1],vals[:,0],label=label,alpha=.5,s=10)
    ax.set_xlabel('DC energy ratio');ax.set_ylabel('Static-to-total ratio');ax.legend();fig.tight_layout()
    fig.savefig(out/'fig6_static_ratio_reconstructed.png',dpi=200);plt.close(fig)
    # Representative images, matching plot intent but no matched frames assertion
    rows=[('RT',sim),('CDR',cdr),('Real',real)]
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,(label,seq) in zip(axes,rows):
        ax.imshow(image_rgb(seq[0],*bounds,224).transpose(1,2,0));ax.set_title(label);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'fig5_rd_montage_reconstructed.png',dpi=200);plt.close(fig)
    (out/'figure_note.txt').write_text('These are reconstructed diagnostics, NOT the original published figure outputs.\n')
    return stats
