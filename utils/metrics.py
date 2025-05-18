import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_auc(scores, labels):
    return roc_auc_score(labels, scores)