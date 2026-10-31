"""Compare measured new run to target paper metrics without rewriting results."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics-json',required=True);p.add_argument('--task',choices=['binary','counting'],required=True)
    p.add_argument('--targets',default=str(Path(__file__).resolve().parents[2]/'configs'/'published_targets.json'))
    a=p.parse_args();metrics=json.loads(Path(a.metrics_json).read_text());target=json.loads(Path(a.targets).read_text())
    ref=target['binary_occupancy' if a.task=='binary' else 'three_class_counting']
    for name,key in [('accuracy','cdr_accuracy_from_confusion' if a.task=='binary' else 'cdr_accuracy'),('balanced_accuracy','cdr_balanced_accuracy_from_confusion' if a.task=='binary' else 'cdr_balanced_accuracy')]:
        measured=float(metrics[name]);gold=float(ref[key])
        print(f'{name}: observed={measured:.8f}; archived={gold:.8f}; difference={measured-gold:+.8f}')
    got=np.array(metrics['confusion_matrix']);old=np.array(ref['cdr_confusion_rows_true_cols_pred'])
    print('Same exact confusion matrix?',np.array_equal(got,old))
    print('Observed class supports',got.sum(axis=1).tolist(),'paper',ref['test_support'])
    print('Not a reproduction certificate: exact weights / preprocessing and dataset provenance require verification')
if __name__=='__main__':main()
