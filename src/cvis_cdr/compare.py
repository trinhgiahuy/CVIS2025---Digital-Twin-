"""Evaluate original-style three methods and save prediction caches + plots.
Select checkpoints by SIM validation ONLY; test scores are descriptive.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from .experiment import RadarDataset,model_init,data_rows,evaluate_model
from .records import read_manifest
from .figures import save_confusion

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--task',choices=['binary','counting'],default='counting');p.add_argument('--seed',type=int,default=42)
 p.add_argument('--artifacts-dir',default='artifacts_3_class');p.add_argument('--out');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='auto');a=p.parse_args()
 out=Path(a.out) if a.out else Path(a.artifacts_dir)/a.task/'eval_cache'
 out.mkdir(parents=True,exist_ok=True)
 device=torch.device(a.device if a.device!='auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
 df=read_manifest(a.manifest)
 report={}; cms=[]
 for method in ['naive','random_dr','cdr']:
  path=Path(a.artifacts_dir)/a.task/f'resnet18_{method}_seed_{a.seed}'/'best_model.pth'
  if not path.exists():raise FileNotFoundError('Checkpoint missing: '+str(path))
  pack=torch.load(path,map_location='cpu',weights_only=True)
  n=pack['nclasses'];ds=RadarDataset(data_rows(df,'real','test',list(range(n))),'naive',pack['bounds'],pack['input_size'],pack['seed'],pack['calibration'],pack.get('window','none'),pack.get('dc_remove',False))
  net=model_init(n).to(device);net.load_state_dict(pack['model'])
  met,y,p,proba=evaluate_model(net,ds,a.batch_size,device,n)
  np.savez_compressed(out/f'resnet18_{method}_seed{a.seed}_preds.npz',y_true=y,y_pred=p,prob=proba)
  save_confusion(met['confusion_matrix'],out/f'{method}_cm.png',method)
  report[method]={k:met[k] for k in ['accuracy','balanced_accuracy','macro_f1','support','confusion_matrix']}
  cms.append(np.array(met['confusion_matrix']))
  print(method,report[method])
 for i,cm in enumerate(cms[1:]):
  if cm.sum()!=cms[0].sum():raise ValueError('Different test set sizes across methods')
  if not np.array_equal(np.sum(cm,axis=1),np.sum(cms[0],axis=1)):
   raise ValueError('Class test supports differ across methods')
 (out/'compare_metrics.json').write_text(json.dumps(report,indent=2))
 fig,axes=plt.subplots(1,3,figsize=(11,3.4))
 for ax,m,cm in zip(axes,report,cms):
  ax.imshow(cm,cmap='Blues');ax.set_title(m+'\n'+f"bal={report[m]['balanced_accuracy']:.3f}")
  ax.set_xlabel('predicted');ax.set_ylabel('true')
  for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=9)
 fig.tight_layout();fig.savefig(out/'figure_8_or_9_comparison_reconstructed.png',dpi=200);plt.close(fig)
 print('Saved',out)
if __name__=='__main__':main()
