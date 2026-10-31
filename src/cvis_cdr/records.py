"""Manifest-driven data loading. NO guess that a file is FFT input.

CSV schema: path,domain,label,split,kind,session_id,participant_id
- domain: sim|real
- label: 0(empty),1(one person),2(two people); calibration may use -1
- split: train|val|test|calibration
- kind: rd_db|rd_linear|rd_complex|adc_iq_json|adc_complex

Path can be .csv, .txt, .npy, .npz, .json, .h5/.hdf5. For complex
arrays, use .npy/.npz; for JSON define {"real": [[...]], "imag": [[...]]}.
HDF5 path supports /dataset notation via path::/dataset.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from .processing import adc_to_rd_db,magnitude_to_db

NEEDED=("path","domain","label","split","kind","session_id")
ALLOWED_SPLITS={"train","val","test","calibration"}
ALLOWED_KINDS={"rd_db","rd_linear","rd_complex","adc_iq_json","adc_complex"}


def read_manifest(path: str|Path) -> pd.DataFrame:
    path=Path(path).resolve()
    df=pd.read_csv(path,keep_default_na=False)
    miss=set(NEEDED)-set(df.columns)
    if miss: raise ValueError(f"Manifest missing {sorted(miss)}")
    df["label"]=pd.to_numeric(df["label"],errors="raise").astype(int)
    if not set(df["split"]).issubset(ALLOWED_SPLITS):
        raise ValueError(f"Unsupported split values: {set(df['split'])-ALLOWED_SPLITS}")
    if not set(df["kind"]).issubset(ALLOWED_KINDS):
        raise ValueError(f"Unsupported data kinds: {set(df['kind'])-ALLOWED_KINDS}")
    def make_path(item):
        raw=str(item)
        if "::" in raw:
            outer, inner=raw.split("::",1)
            return str((path.parent / outer).resolve() if not Path(outer).is_absolute() else Path(outer)) + "::" + inner
        return str((path.parent / raw).resolve() if not Path(raw).is_absolute() else Path(raw))
    df["path"]=[make_path(v) for v in df["path"]]
    if not len(df):raise ValueError("Empty manifest")
    if ((df.domain=="real")&(df.split.isin(["train","val"]))).any():
        raise ValueError("The publication trains only on simulation. Real train/val labels are disallowed.")
    if ((df.split=="test")&(df.domain!="real")).any():
        raise ValueError("Evaluation test must be real, not simulated")
    if ((df.split=="calibration")&((df.domain!="real")|(~df.label.isin([-1,0])))).any():
        raise ValueError("Calibration must be real empty-room unlabeled (label -1 or 0)")
    # No identical files between calibration/train/test; record-level duplicate protection.
    dups=df[df.path.duplicated(keep=False)]
    if not dups.empty:
        raise ValueError("Duplicate data path(s) across manifest rows: "+repr(dups.path.tolist()[:6]))
    return df


def read_array(path: str, kind: str, window: str="none", dc_remove: bool=False,
               chirp_axis:int=0,sample_axis:int=1,rx_axis:int|None=2, receiver:int|None=0) -> np.ndarray:
    s=str(path)
    if "::" in s:
        file,dataset=s.split("::",1)
        with h5py.File(file,"r") as f:
            x=f[dataset][()]
    else:
        p=Path(s)
        ext=p.suffix.lower()
        if ext in {".csv",".txt"}:
            x=np.loadtxt(p,delimiter="," if ext==".csv" else None)
        elif ext==".npy":
            x=np.load(p,allow_pickle=False)
        elif ext==".npz":
            with np.load(p,allow_pickle=False) as f:
                if kind=="adc_iq_json": raise ValueError("adc_iq_json must be JSON")
                key="rd_db" if "rd_db" in f else "adc" if "adc" in f else "array" if "array" in f else list(f.keys())[0]
                x=f[key].copy()
        elif ext in {".h5",".hdf5"}:
            raise ValueError("Use path/to/file.h5::/exact/dataset/key; do not guess H5 schema")
        elif ext==".json":
            d=json.loads(p.read_text())
            if not isinstance(d,dict) or "real" not in d or "imag" not in d:
                raise ValueError("JSON I/Q must have real/imag keys; inspect your vendor JSON and adapt explicitly")
            x=np.asarray(d["real"],np.float32)+1j*np.asarray(d["imag"],np.float32)
        else:
            raise ValueError(f"Unsupported extension {ext} for {path}")
    if kind=="rd_db":
        if np.iscomplexobj(x): raise ValueError("rd_db must be real; select rd_complex")
        out=np.asarray(x,np.float32)
    elif kind in {"rd_linear","rd_complex"}:
        out=magnitude_to_db(x)
    elif kind in {"adc_iq_json","adc_complex"}:
        out=adc_to_rd_db(x,chirp_axis,sample_axis,rx_axis,window,dc_remove,receiver=receiver)
    else: raise ValueError(kind)
    if out.ndim!=2 or min(out.shape)<2 or not np.isfinite(out).all():
        raise ValueError(f"Invalid RD result shape={out.shape} from {path}")
    return out


def validate_manifest(df:pd.DataFrame)->dict:
    out={"counts":{},"warnings":[]}
    for (d,s,y),part in df.groupby(["domain","split","label"]):
        out["counts"][f"{d}/{s}/{y}"]=len(part)
    # Avoid session overlap in real calibration and real testing
    real=df[df.domain=="real"]
    a=set(real[real.split=="calibration"].session_id)-{""}
    b=set(real[real.split=="test"].session_id)-{""}
    if a&b: raise ValueError(f"Calibration/test share session_ids: {sorted(a&b)[:5]}")
    sim=df[df.domain=="sim"]
    sim_train=set(sim[sim.split=="train"].session_id)-{""}
    sim_val=set(sim[sim.split=="val"].session_id)-{""}
    if sim_train&sim_val:
        raise ValueError("Simulated train/validation share session/scene identifiers: "+repr(sorted(sim_train&sim_val)[:6]))
    if (sim.session_id=="").any():
        out["warnings"].append("Missing simulation session/scene IDs: adjacent simulated frames may leak between train and validation")
    if len(real[real.split=="calibration"])!=100:
        out["warnings"].append("Paper described ~100 calibration frames; manifest has "+str(len(real[real.split=="calibration"])))
    if (real.session_id=="").any():
        out["warnings"].append("Missing real session_id: cannot independently verify calibration/test separation or number of sessions")
    if "participant_id" not in real or (real.get("participant_id",pd.Series(dtype=str))=="").any():
        out["warnings"].append("Missing real participant_id: count of actual participants UNKNOWN")
    if not len(df[(df.domain=="sim")&(df.split=="train")]):
        raise ValueError("Missing simulation train rows")
    if not len(df[(df.domain=="sim")&(df.split=="val")]):
        raise ValueError("Missing simulation validation rows")
    if not len(df[(df.domain=="real")&(df.split=="test")]):
        raise ValueError("Missing real labeled test rows")
    if not len(df[(df.domain=="real")&(df.split=="calibration")]):
        raise ValueError("Missing real empty-room calibration rows")
    return out
