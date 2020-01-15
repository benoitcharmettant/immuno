from numpy import array, int64
from sklearn.metrics import accuracy_score

# TODO: clean function to make it work for any number of classes

def accuracy_from_logits(logits, y, threshold=0.5):

    assert len(logits) == len(y)

    y_preds = array(logits)
    y = array(y)
    y_preds = (y_preds > threshold).astype(int64)
    
    if len(y_preds[0]) == 1:
        logits = logits.reshape((-1))
        y = y.reshape((-1))
        return accuracy_score(y_preds, y)
    
    if len(y_preds[0]) == 2:
        
        return accuracy_score(y_preds[:,0], y[:,0]), accuracy_score(y_preds[:,1], y[:,1])

    
