"""Compatibility launcher: python -m src3_frame.evaluate_cnn_master_mod ...
Redirects to the reconstructed evaluator CLI with --checkpoint, --manifest.
"""
from cvis_cdr.experiment import main
if __name__=='__main__': main()
