"""Paths reconstructed from previously validated result records.
This module is for compatibility/reference, not proof original bytes were recovered.
"""
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent.parent
PROCESSED_DIR=PROJECT_ROOT/'processed_data_3_class'
ARTIFACTS_DIR=PROJECT_ROOT/'artifacts_3_class'
TEST_H5=PROCESSED_DIR/'test_final.h5'
NORM_STATS=PROCESSED_DIR/'norm_stats.json'
IMAGE_SIZE=(224,224)
