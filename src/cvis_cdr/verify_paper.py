"""Verify archival CONFUSION MATRICES numerically; not re-executing original checkpoint."""
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

PUBLISHED_COUNTING_CDR=np.array([[423,4,0],[3,88,93],[2,45,106]])
PUBLISHED_BINARY_CDR=np.array([[417,10],[8,176]])
PUBLISHED_COUNTING_NAIVE=np.array([[0,427,0],[0,184,0],[0,153,0]])

def metrics_from_cm(cm):
    y=np.repeat(np.arange(cm.shape[0]),cm.sum(axis=1)); p=np.hstack([np.repeat(np.arange(cm.shape[1]),cm[i]) for i in range(cm.shape[0])])
    return {"accuracy":accuracy_score(y,p),"balanced_accuracy":balanced_accuracy_score(y,p),"macro_f1":f1_score(y,p,average='macro',zero_division=0)}

if __name__=='__main__':
    for label,cm in [('counting_cdr',PUBLISHED_COUNTING_CDR),('binary_cdr',PUBLISHED_BINARY_CDR)]:
        print(label,metrics_from_cm(cm))
