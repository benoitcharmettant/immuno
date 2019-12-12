import torch.nn as nn
from torch.optim import Adam
from numpy import mean

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

        return self.loss(y_pred, y_gt)

    def train_epoch(self, optimizer, train_loader):

        self.train()

        losses = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.float().to(self.device)

            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.get_loss(y_pred, y)

            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().item())
        return mean(losses)

    def start_training(self, train_loader, val_loader, epoch=20, lr=0.01, logger=None):
        self.to(self.device)

        # TODO: add options for the optimizer !
        optimizer = Adam(self.parameters(), lr=lr)

        smooth_loss = self.train_epoch(optimizer, train_loader)

        for e in range(epoch - 1):
            # train phase
            loss_train = self.train_epoch(optimizer, train_loader)
            smooth_loss = 0.99 * smooth_loss + 0.01 * loss_train

            # validation phase

            loss_eval = self.start_eval(val_loader)

            my_print(
                'Train Epoch: {}/{}\tLoss: {:.6f}\t (Eval Loss: {:.6f})'.format(e + 1, epoch, smooth_loss, loss_eval),
                logger=logger)

    def start_eval(self, eval_loader):
        self.eval()
        losses = []
        for batch_idx, (x, y) in enumerate(eval_loader):
            x, y = x.to(self.device), y.float().to(self.device)
            y_pred = self.forward(x)
            loss = self.get_loss(y_pred, y)

            losses.append(loss.cpu().item())

        return mean(losses)
