"""Generate synthetic toy radar-shaped matrices to TEST the reconstruction.
Not physics-grounded, not training evidence, not a replacement for actual data.
"""
import argparse,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='toy_data');p.add_argument('--calibration-frames',type=int,default=8);p.add_argument('--train-per-class',type=int,default=8);p.add_argument('--val-per-class',type=int,default=3);p.add_argument('--test-per-class',type=int,default=5);a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(88)
    rows=[]
    for domain,split,n in [('sim','train',a.train_per_class),('sim','val',a.val_per_class),('real','test',a.test_per_class),('real','calibration',a.calibration_frames)]:
        labels=[0] if split=='calibration' else [0,1,2]
        for label in labels:
            for i in range(n):
                shape=(128,128) if domain=='sim' else (128,256)
                bg=-155 if domain=='sim' else -114
                x=rng.normal(bg,3.0 if domain=='sim' else 6.0,shape).astype('float32')
                if label:
                    x[28:42,40:50]=rng.normal(-75,3,(14,10))
                    if label==2:x[75:85,65:80]=rng.normal(-78,3,(10,15))
                path=root/f'{domain}_{split}_{label}_{i:04}.npy';np.save(path,x)
                rows.append({'path':str(path.resolve()),'domain':domain,'label':-1 if split=='calibration' else label,'split':split,'kind':'rd_db','session_id':f'{domain}_{split}_{label}_{i//2}','participant_id':f'FAKE_{label}' if label>0 else ''})
    # calibration sessions have distinct identifiers from all testing sessions
    manifest=root/'manifest.csv'
    with open(manifest,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('TOY DATA ONLY, NOT PAPER DATA:',manifest,'rows',len(rows))
if __name__=='__main__':main()
