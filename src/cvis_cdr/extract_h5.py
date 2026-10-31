"""Explicit-key H5 frame extractor. It never guesses which H5 dataset is dB.

Example: python -m cvis_cdr.extract_h5 data.h5 --frames-key /rd_db --labels-key /labels --split test --domain real --out data/real_from_h5

The output manifest should be merged with simulation+calibration rows manually.
Do not use this for original source h5 without inspecting keys and value ranges.
"""
import argparse,csv
from pathlib import Path
import numpy as np,h5py

def main():
    p=argparse.ArgumentParser();p.add_argument('h5');p.add_argument('--frames-key',required=True);p.add_argument('--labels-key')
    p.add_argument('--split',choices=['train','val','test','calibration'],required=True)
    p.add_argument('--domain',choices=['sim','real'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=['rd_db','rd_linear','rd_complex','adc_complex'],default='rd_db')
    p.add_argument('--session-id',default='');p.add_argument('--participant-id',default='');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    with h5py.File(a.h5) as h:
        x=h[a.frames_key]
        if x.ndim!=3 and x.ndim!=4:raise ValueError('Expect (N,H,W) RD or (N,chirps,samples,rx) ADC data')
        labels=h[a.labels_key][()] if a.labels_key else np.full((len(x),),-1 if a.split=='calibration' else -99)
        if len(labels)!=len(x):raise ValueError('labels length differs from N frames')
        if not a.labels_key and a.split!='calibration':raise ValueError('Labels required for test/train/val')
        for i in range(len(x)):
            f=out/f'frame_{i:06d}.npy';np.save(f,x[i])
            rows.append(dict(path=str(f.resolve()),domain=a.domain,label=int(labels[i]),split=a.split,kind=a.kind,
                             session_id=a.session_id,participant_id=a.participant_id))
    dest=out/'manifest_part.csv'
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Exported',len(rows),'samples ->',dest)
    print('IMPORTANT: all frames are tagged with the same session ID unless overridden. Restore original session/participant metadata separately!')
if __name__=='__main__':main()
