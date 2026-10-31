from argparse import Namespace
from pathlib import Path
from cvis_cdr.smoke_data import main as smoke_main
from cvis_cdr.experiment import training,eval_checkpoint
from cvis_cdr.records import read_manifest,validate_manifest
import sys

def test_minimal_training(tmp_path,monkeypatch):
    root=tmp_path/'data'
    monkeypatch.setattr(sys,'argv',['smoke_data','--root',str(root),'--train-per-class','2','--val-per-class','2','--test-per-class','2','--calibration-frames','3'])
    smoke_main()
    manifest=root/'manifest.csv'
    assert validate_manifest(read_manifest(manifest))['counts']['real/test/2']==2
    args=Namespace(command='train',manifest=str(manifest),output_dir=str(tmp_path/'results'),task='binary',method='cdr',seed=40,epochs=1,patience=1,lr=.0001,weight_decay=.0001,batch_size=4,input_size=64,vmin=-200.,vmax=-50.,norm_stats=None,window='none',dc_remove=False,device='cpu')
    path=training(args)
    assert (path/'best_model.pth').exists()
    assert (path/'test_metrics.json').exists()
    eval_checkpoint(Namespace(manifest=str(manifest),checkpoint=str(path/'best_model.pth'),output_dir=str(tmp_path/'eval'),batch_size=4,device='cpu'))
    assert (tmp_path/'eval'/'metrics.json').exists()
