import csv
from pathlib import Path
import numpy as np
import h5py
from cvis_cdr.records import read_array,read_manifest

def test_h5_subdataset_relative_path(tmp_path):
    f=tmp_path/'samples.h5'
    with h5py.File(f,'w') as h:h.create_dataset('frames/frame0',data=np.ones((12,20),dtype=np.float32)*-115)
    rows=[{'path':'samples.h5::/frames/frame0','domain':'real','split':'calibration','label':'-1','kind':'rd_db','session_id':'cal0','participant_id':''}]
    manifest=tmp_path/'manifest.csv'
    with manifest.open('w',newline='') as fo:
        w=csv.DictWriter(fo,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    df=read_manifest(manifest)
    x=read_array(df.iloc[0]['path'],df.iloc[0]['kind'])
    assert x.shape==(12,20) and float(x.mean())==-115
