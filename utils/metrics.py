from numpy import array, int64
from sklearn.metrics import accuracy_score


def accuracy_from_logits(logits, y, threshold=0.5):
    assert len(logits) == len(y)

    nb_classes = len(logits[0])

    y_preds = array(logits)
    y = array(y)
    y_preds = (y_preds > threshold).astype(int64)

    accuracies = []

    for i in range(nb_classes):
        accuracies.append(accuracy_score(y[:, i], y_preds[:, i]))

    return accuracies
