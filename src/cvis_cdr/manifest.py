"""Create an editable manifest from a folder of per-frame .csv/.npy/.npz files.
Folder convention: DATA_ROOT/{sim_train,sim_val,real_test,real_calibration}/{empty,one,two}/*
Data kind MUST be confirmed by user; default rd_db assumes files ALREADY hold RD in dB.
Filename 'session' metadata cannot be reliably guessed -> placeholder session_id.
"""
from pathlib import Path
import argparse,csv
from .records import read_manifest,validate_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',default='manifest.csv')
    p.add_argument('--kind',choices=['rd_db','rd_linear','rd_complex','adc_complex','adc_iq_json'],default='rd_db')
    a=p.parse_args();root=Path(a.root).resolve()
    mapping={'sim_train':('sim','train'),'sim_val':('sim','val'),
             'real_test':('real','test'),'real_calibration':('real','calibration')}
    labels={'empty':0,'one':1,'two':2}
    rows=[]
    for folder,(domain,split) in mapping.items():
        for class_name,label in labels.items():
            if split=='calibration' and label!=0:continue
            loc=root/folder/class_name
            if not loc.exists():continue
            for f in sorted(loc.rglob('*')):
                if f.is_file() and f.suffix.lower() in {'.csv','.npy','.npz','.json'}:
                    # Directory immediate below class = session if present, else unknown.
                    relative=f.relative_to(loc)
                    sess=relative.parts[0] if len(relative.parts)>1 else ''
                    rows.append({'path':str(f),'domain':domain,'label':label if split!='calibration' else -1,
                                 'split':split,'kind':a.kind,'session_id':sess,'participant_id':''})
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='') as fo:
        writer=csv.DictWriter(fo,fieldnames=['path','domain','label','split','kind','session_id','participant_id'])
        writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} records to {dest}. CHECK kind, path, group metadata, and separation before training.')
    if rows:
        d=read_manifest(dest)
        try:print(validate_manifest(d))
        except ValueError as e: print('MANIFEST REQUIRES ATTENTION:',e)
if __name__=='__main__':main()
