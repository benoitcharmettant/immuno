import torch.nn as nn
from torch.optim import Adam
from numpy import mean
from sklearn.metrics import accuracy_score

from utils.tools import my_print


class Model(nn.Module):
    def __init__(self, input_shape, loss, device="cuda:0"):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.loss = loss
        self.device = device

    def forward(self, x):
        pass

    def get_loss(self, y_pred, y_gt):
        y_pred = y_pred.reshape((-1))

        return self.loss(y_pred, y_gt)

    def train_epoch(self, optimizer, train_loader):

        self.train()

        losses = []

        metrics = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.float().to(self.device)

            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.get_loss(y_pred, y)


            loss.backward()
            optimizer.step()

            metric = accuracy_score(y.cpu().detach().numpy() > 0.5, y_pred.cpu().detach().numpy())

            losses.append(loss.cpu().item())
            metrics.append(metric)
        return mean(losses), mean(metrics)

    def start_training(self, train_loader, val_loader, epoch=20, lr=0.01, logger=None, regularization=0):
        self.to(self.device)
        # TODO: add modular metrics system !
        # TODO: add options for the optimizer !
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=regularization)

        smooth_loss, _ = self.train_epoch(optimizer, train_loader)
        smooth_loss_val, _ = self.start_eval(val_loader)

        for e in range(epoch - 1):
            # train phase
            loss_train, metric_train = self.train_epoch(optimizer, train_loader)
            smooth_loss = 0.99 * smooth_loss + 0.01 * loss_train

            # validation phase

            loss_val, metric_val = self.start_eval(val_loader)

            smooth_loss_val = 0.99 * smooth_loss_val + 0.01 * loss_val

            my_print(
                'Train Epoch: {}/{}\tLoss: {:.6f} - Acc: {:.3f}\t(Eval Loss: {:.6f} - Acc: {:.3f})'.format(e + 1,
                                                                                                           epoch,
                                                                                                           smooth_loss,
                                                                                                           metric_train,
                                                                                                           smooth_loss_val,
                                                                                                           metric_val),
                logger=logger)

    def start_eval(self, eval_loader):
        self.eval()
        losses = []
        metrics = []
        for batch_idx, (x, y) in enumerate(eval_loader):
            x, y = x.to(self.device), y.float().to(self.device)
            y_pred = self.forward(x)
            loss = self.get_loss(y_pred, y)
            metric = accuracy_score(y.cpu().detach().numpy() > 0.5, y_pred.cpu().detach().numpy())

            losses.append(loss.cpu().item())
            metrics.append(metric)

        return mean(losses), mean(metrics)
