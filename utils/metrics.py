from numpy import array, int64
from sklearn.metrics import accuracy_score


def accuracy_from_logits(logits, y, threshold=0.5):
    
    logits = logits.reshape((-1))
    y = y.reshape((-1))

    assert len(logits) == len(y)

    y_preds = array(logits)
    y_preds = (y_preds > threshold).astype(int64)

    return accuracy_score(y_preds, y)
