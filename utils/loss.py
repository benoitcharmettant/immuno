import torch
import torch.nn as nn


def get_loss_function(loss_type):
    if loss_type == 'bce':
        loss_fun = nn.BCELoss()

    else:
        raise Exception('Undifined loss function!')

    return loss_fun
