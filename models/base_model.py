from os.path import join

from torch import save
import torch.nn as nn
from torch.optim import Adam
from numpy import mean, array, append

from utils.tools import my_print
from utils.metrics import accuracy_from_logits
from utils.visualisation import plot_training


class Model(nn.Module):
    def __init__(self, input_shape, loss, device="cuda:1"):
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

            metric = accuracy_from_logits(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy())

            losses.append(loss.cpu().item())
            metrics.append(metric)
        return mean(losses), mean(metrics)

    def start_training(self, train_loader, val_loader, epoch=20, lr=0.01, logger=None, regularization=0,
                       random_pred_level=None):
        self.to(self.device)
        # TODO: add modular metrics system !
        # TODO: add options for the optimizer !
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=regularization)

        lowest_eval_loss = 1000

        for e in range(epoch - 1):
            # train phase
            loss_train, metric_train = self.train_epoch(optimizer, train_loader)

            # validation phase

            y_pred_val, y_val, loss_val = self.evaluate(val_loader)
            metric_val = accuracy_from_logits(y_pred_val, y_val)

            if loss_val < lowest_eval_loss:
                lowest_eval_loss = loss_val
                weight_path = join(logger.root_dir, "best_model.pth")
                my_print(f"Saving model in {weight_path}", logger=logger)
                save(self, weight_path)

            my_print(
                'Train Epoch: {}/{}\tLoss: {:.6f} - Acc: {:.3f}\t(Eval Loss: {:.6f} - Acc: {:.3f})'.format(e + 1,
                                                                                                           epoch,
                                                                                                           loss_train,
                                                                                                           metric_train,
                                                                                                           loss_val,
                                                                                                           metric_val),
                logger=logger)

            if e > 0 and e % 100 == 0 and logger is not None:
                plot_training(logger.root_dir, random_pred_level=random_pred_level)

    def evaluate(self, val_loader):
        self.eval()

        gt = None
        preds = None
        losses = []

        for (x, y) in val_loader:

            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.forward(x)

            loss = self.get_loss(y_pred.float(), y.float())
            losses.append(loss.cpu().detach().numpy())

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            if preds is not None:
                preds = append(preds, y_pred, axis=0)
            else:
                preds = y_pred

            if gt is not None:
                gt = append(gt, y, axis=0)
            else:
                gt = y

        return preds, gt, mean(losses)
