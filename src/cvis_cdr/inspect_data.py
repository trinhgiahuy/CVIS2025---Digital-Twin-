"""Read-only H5/CSV/NPZ inspection, no schema guessing."""
from pathlib import Path
import argparse,json
import h5py,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');args=p.parse_args()
    for name in args.paths:
        f=Path(name);print('\nFILE',f,'size',f.stat().st_size)
        if f.suffix.lower() in ('.h5','.hdf5'):
            with h5py.File(f) as h:
                def describe(path,x):
                    if isinstance(x,h5py.Dataset):
                        print('  ',path,'shape=',x.shape,'dtype=',x.dtype,'attrs=',dict(x.attrs))
                h.visititems(describe)
                print('ROOT ATTRS:',dict(h.attrs))
        elif f.suffix.lower()=='.npz':
            with np.load(f,allow_pickle=False) as x:
                for k in x:print('  ',k,x[k].shape,x[k].dtype)
        elif f.suffix.lower()=='.npy':
            x=np.load(f,mmap_mode='r',allow_pickle=False);print('  shape=',x.shape,'dtype=',x.dtype)
        elif f.suffix.lower()=='.json':
            d=json.loads(f.read_text());print('  keys=',list(d) if isinstance(d,dict) else type(d))
        else: print('  Format not inspected automatically')
if __name__=='__main__':main()
