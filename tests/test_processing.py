import numpy as np
import pytest
from cvis_cdr.processing import cdr_floor,random_floor,calibration_stats,image_rgb,adc_to_rd_db
from cvis_cdr.records import read_manifest,validate_manifest

def test_cdr_is_db_max():
    src=np.array([[-140.,-90.],[-110.,-100.]],dtype='float32')
    rng=np.random.default_rng(7)
    actual=cdr_floor(src,rng,-115,3)
    expected=np.maximum(src,np.random.default_rng(7).normal(-115,3,src.shape))
    np.testing.assert_allclose(actual,expected,atol=1e-5)
    assert (actual>=src).all()

def test_calibration_pooled():
    a=[np.array([[-110,-120]]),np.array([[-100,-130]])]
    s=calibration_stats(a)
    assert s['mu_db']==-115
    assert abs(s['sigma_db']-np.std([-110,-120,-100,-130]))<1e-8
    assert s['calibration_frames']==2

def test_rgb_shape_and_bounds():
    x=np.linspace(-200,-60,32*16).reshape(32,16)
    y=image_rgb(x,-200,-60,64)
    assert y.shape==(3,64,64) and 0<=y.min()<=y.max()<=1

def test_adc_fft():
    x=np.ones((4,8),dtype='complex64')
    z=adc_to_rd_db(x,rx_axis=None,window='none')
    assert z.shape==(8,4) and np.isfinite(z).all()
