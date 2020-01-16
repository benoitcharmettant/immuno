from numpy import array, int64, mean
from sklearn.metrics import accuracy_score,roc_auc_score
import pandas as pd

def calculate_metric(logits, y, threshold=0.5):

    metric = {}
    metric['accuracy']= accuracy_from_logits(logits, y, threshold)
    metric['auc']= calculate_auc(logits, y)

    return metric

def accuracy_from_logits(logits, y, threshold=0.5):
    assert len(logits) == len(y)

    nb_classes = len(logits[0])

    y_preds = array(logits)
    y = array(y)
    y_preds = (y_preds > threshold).astype(int64)

    accuracies ={}

    for i in range(nb_classes):
        accuracies[i]=accuracy_score(y[:, i], y_preds[:, i])

    y_preds = y_preds.reshape((-1))
    y = y.reshape((-1))
    accuracies['all']=accuracy_score(y, y_preds)
    return accuracies

def calculate_auc(logits, y):
    assert len(logits) == len(y)

    nb_classes = len(logits[0])

    y_preds = array(logits)
    y = array(y)

    AUCs = {}

    for i in range(nb_classes):
        AUCs[i] = get_roc_auc_score(y[:, i], y_preds[:, i])

    y_preds = y_preds.reshape((-1))
    y = y.reshape((-1))
    AUCs['all'] = get_roc_auc_score(y, y_preds)
    return AUCs

def get_roc_auc_score(y,y_preds):
    try:
        auc=roc_auc_score(y, y_preds)
    except ValueError:
        auc=0

    return auc



def calculate_mean(metric_dic):

    final_mean_metric={}
    for metric_type in metric_dic.keys():
        metric=metric_dic[metric_type]
        metric=pd.DataFrame(metric)
        mean_metric=metric.mean()
        final_mean_metric[metric_type]=mean_metric.to_dict()


    return final_mean_metric