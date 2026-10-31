"""Sim-only ResNet18 training and held-out real evaluation.
Independent reconstruction; exact lost code/hyperparameters are unknown.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,classification_report
from .records import read_manifest,read_array,validate_manifest
from .processing import image_rgb,calibration_stats,cdr_floor,random_floor


def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def model_init(nclasses:int):
    net=resnet18(weights=None)  # mandatory: random initial weights, not ImageNet
    net.fc=nn.Linear(net.fc.in_features,nclasses)
    return net


def data_rows(df,domain,split,labels):
    return df[(df.domain==domain)&(df.split==split)&(df.label.isin(labels))].to_dict('records')

class RadarDataset(Dataset):
    def __init__(self, rows, method, clip, size, seed, calib, window,dc_remove, cache=True):
        self.rows=list(rows);self.method=method;self.clip=clip;self.size=size
        self.seed=seed;self.calib=calib;self.window=window;self.dc_remove=dc_remove
        self.epoch=0
        self.arr=[read_array(r['path'],r['kind'],window=window,dc_remove=dc_remove) for r in self.rows] if cache else None
    def __len__(self):return len(self.rows)
    def set_epoch(self,epoch):self.epoch=epoch
    def __getitem__(self,idx):
        rec=self.rows[idx]
        s=self.arr[idx] if self.arr is not None else read_array(rec['path'],rec['kind'],window=self.window,dc_remove=self.dc_remove)
        if self.method!='naive':
            rng=np.random.default_rng(self.seed+104729*self.epoch+8191*idx)
            if self.method=='random_dr':s=random_floor(s,rng)
            elif self.method=='cdr':s=cdr_floor(s,rng,self.calib['mu_db'],self.calib['sigma_db'])
            else:raise ValueError(self.method)
        rgb=image_rgb(s,self.clip[0],self.clip[1],self.size)
        return torch.from_numpy(rgb.copy()),int(rec['label'])


def choose_bounds(df,args):
    if args.vmin is not None or args.vmax is not None:
        if args.vmin is None or args.vmax is None:raise ValueError("Specify BOTH --vmin and --vmax")
        return [float(args.vmin),float(args.vmax)],"explicit_cli"
    if args.norm_stats:
        obj=json.loads(Path(args.norm_stats).read_text())
        # Cannot guess JSON's meaning; limit to well-defined names.
        for a,b in [('vmin','vmax'),('min','max'),('clip_vmin','clip_vmax')]:
            if a in obj and b in obj:
                return [float(obj[a]),float(obj[b])],f"file:{args.norm_stats}:{a},{b}"
        raise ValueError("norm_stats JSON lacks recognized numeric (vmin,vmax), (min,max), or (clip_vmin,clip_vmax)")
    bounds=[]
    for r in data_rows(df,'sim','train',[0,1,2]):
        x=read_array(r['path'],r['kind'],window=args.window,dc_remove=args.dc_remove)
        bounds.extend([float(np.min(x)),float(np.max(x))])
    if not bounds:raise ValueError('No simulated samples for computing normalization bounds')
    return [min(bounds),max(bounds)],'simulation_train_minmax_reconstructed'


def prepare(df,args,method,labels,seed):
    validate_manifest(df)
    calrows=data_rows(df,'real','calibration',[-1,0])
    # Calibration stats computed only from unlabeled real empty-room data.
    frames=[read_array(r['path'],r['kind'],window=args.window,dc_remove=args.dc_remove) for r in calrows]
    calib=calibration_stats(frames)
    bounds,source=choose_bounds(df,args)
    ds={}
    for split in ['train','val']:
        ds[split]=RadarDataset(data_rows(df,'sim',split,labels),method if split=='train' else 'naive',bounds,args.input_size,seed,calib,args.window,args.dc_remove, cache=not getattr(args,'no_cache',False))
        if not len(ds[split]):raise ValueError(f'No {split} simulation samples for labels {labels}')
    ds['test']=RadarDataset(data_rows(df,'real','test',labels),'naive',bounds,args.input_size,seed,calib,args.window,args.dc_remove, cache=not getattr(args,'no_cache',False))
    if not len(ds['test']):raise ValueError('No real held-out test samples')
    return ds,calib,bounds,source


def evaluate_model(model,ds,batch_size,device,nclasses):
    ld=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval();true=[];pred=[];proba=[]
    with torch.inference_mode():
        for x,y in ld:
            logits=model(x.to(device));p=torch.softmax(logits,1).cpu().numpy()
            proba.extend(p);pred.extend(np.argmax(p,axis=1).tolist());true.extend(y.numpy().tolist())
    met={"accuracy":float(accuracy_score(true,pred)),
         "balanced_accuracy":float(balanced_accuracy_score(true,pred)),
         "macro_f1":float(f1_score(true,pred,average='macro',zero_division=0)),
         "confusion_matrix":confusion_matrix(true,pred,labels=list(range(nclasses))).tolist(),
         "support":{str(k):int(sum(v==k for v in true)) for k in range(nclasses)},
         "classification_report":classification_report(true,pred,labels=list(range(nclasses)),output_dict=True,zero_division=0)}
    return met,np.asarray(true),np.asarray(pred),np.asarray(proba)


def training(args):
    seed_all(args.seed)
    df=read_manifest(args.manifest)
    nclasses=2 if args.task=='binary' else 3
    labels=list(range(nclasses))
    if args.method not in ['naive','random_dr','cdr']:raise ValueError(args.method)
    ds,calib,bounds,source=prepare(df,args,args.method,labels,args.seed)
    print('Data counts train/val/test',*[len(ds[k]) for k in ['train','val','test']])
    print('Calibration mu/sigma',calib,'Normalization bounds',bounds,source)
    out=Path(args.output_dir)/args.task/f'resnet18_{args.method}_seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device if args.device!='auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    model=model_init(nclasses).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion=nn.CrossEntropyLoss()
    best=-float('inf');best_epoch=-1;stale=0
    trainrows=[]
    for epoch in range(args.epochs):
        ds['train'].set_epoch(epoch)
        dl=DataLoader(ds['train'],batch_size=args.batch_size,shuffle=True,num_workers=0)
        model.train();running=0.;seen=0
        for x,y in dl:
            x=x.to(device);y=y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(x);loss=criterion(logits,y)
            loss.backward();opt.step()
            running+=float(loss.item())*len(y);seen+=len(y)
        val,_,_,_=evaluate_model(model,ds['val'],args.batch_size,device,nclasses)
        row={'epoch':epoch+1,'train_loss':running/max(seen,1),'sim_val_bal_acc':val['balanced_accuracy'],'sim_val_acc':val['accuracy']}
        trainrows.append(row)
        print(json.dumps(row),flush=True)
        if val['balanced_accuracy']>best+1e-8:
            best=val['balanced_accuracy'];best_epoch=epoch+1;stale=0
            torch.save({'model':model.state_dict(),'task':args.task,'method':args.method,'seed':args.seed,
                       'nclasses':nclasses,'bounds':bounds,'bounds_source':source,'calibration':calib,
                       'input_size':args.input_size,'window':args.window,'dc_remove':args.dc_remove,
                       'train_cfg':vars(args),'best_sim_val_bal_acc':best,'best_epoch':best_epoch,
                       'manifest_sha256':file_hash(args.manifest)},out/'best_model.pth')
        else:
            stale+=1
            if stale>=args.patience:break
    with open(out/'train_log.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(trainrows[0]));wr.writeheader();wr.writerows(trainrows)
    # Test exactly once, after simulated validation chooses checkpoint. Never test per epoch.
    pack=torch.load(out/'best_model.pth',map_location=device,weights_only=True)
    model.load_state_dict(pack['model'])
    test,true,pred,prob=evaluate_model(model,ds['test'],args.batch_size,device,nclasses)
    (out/'test_metrics.json').write_text(json.dumps(test,indent=2))
    np.savez_compressed(out/'test_predictions.npz',y_true=true,y_pred=pred,prob=prob)
    (out/'run_meta.json').write_text(json.dumps({'paper_result_NOT_guaranteed':True,'checkpoint':str(out/'best_model.pth'),
            'test':test,'train_count':len(ds['train']),'sim_val_count':len(ds['val']),
            'test_count':len(ds['test']),'calibration':calib,'bounds':bounds,'bounds_source':source,
            'manifest_sha256':pack['manifest_sha256'],'best_epoch':best_epoch},indent=2))
    from .figures import save_confusion
    save_confusion(test['confusion_matrix'],out/'test_confusion.png',f'{args.task} {args.method} seed {args.seed}')
    print('Test results:',json.dumps({k:test[k] for k in ['accuracy','balanced_accuracy','macro_f1','support']},indent=2))
    print('Saved:',out.resolve())
    return out


def eval_checkpoint(args):
    from .records import read_manifest
    df=read_manifest(args.manifest)
    pack=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    bounds=pack['bounds'];nclasses=int(pack['nclasses'])
    labels=list(range(nclasses))
    rows=data_rows(df,'real','test',labels)
    if not rows: raise ValueError('No matching real test rows')
    ds=RadarDataset(rows,'naive',bounds,int(pack['input_size']),int(pack['seed']),pack['calibration'],
                    pack.get('window','none'),pack.get('dc_remove',False))
    device=torch.device(args.device if args.device!='auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    model=model_init(nclasses).to(device);model.load_state_dict(pack['model'])
    test,y,p,prob=evaluate_model(model,ds,args.batch_size,device,nclasses)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(test,indent=2))
    np.savez_compressed(out/'predictions.npz',y_true=y,y_pred=p,prob=prob)
    from .figures import save_confusion
    save_confusion(test['confusion_matrix'],out/'confusion.png',f"{pack['task']} {pack['method']} seed {pack['seed']}")
    print(json.dumps({k:test[k] for k in ['accuracy','balanced_accuracy','macro_f1','confusion_matrix','support']},indent=2))


def cli_parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    def common(q):
        q.add_argument('--manifest',required=True)
        q.add_argument('--output-dir',default='artifacts_3_class')
        q.add_argument('--batch-size',type=int,default=16)
        q.add_argument('--device',default='auto')
    t=sub.add_parser('train');common(t)
    t.add_argument('--task',choices=['binary','counting'],required=True)
    t.add_argument('--method',choices=['naive','random_dr','cdr'],required=True)
    t.add_argument('--seed',type=int,default=42)
    t.add_argument('--epochs',type=int,default=30)
    t.add_argument('--patience',type=int,default=7)
    t.add_argument('--lr',type=float,default=0.0001)
    t.add_argument('--weight-decay',type=float,default=0.0001)
    t.add_argument('--input-size',type=int,default=224)
    t.add_argument('--vmin',type=float)
    t.add_argument('--vmax',type=float)
    t.add_argument('--norm-stats',help='Optional original norm_stats.json; recognized keys min/max or vmin/vmax')
    t.add_argument('--window',choices=['none','hann','blackmanharris'],default='none')
    t.add_argument('--dc-remove',action='store_true')
    t.add_argument('--no-cache',action='store_true',help='Load per-frame arrays on demand; avoid holding large datasets in RAM')
    e=sub.add_parser('evaluate');common(e);e.add_argument('--checkpoint',required=True)
    return p


def main():
    args=cli_parser().parse_args()
    if args.command=='train':training(args)
    else:eval_checkpoint(args)
if __name__=='__main__':main()
