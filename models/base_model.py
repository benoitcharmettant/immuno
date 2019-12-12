import torch.nn as nn
from torch.optim import SGD, Adam
from torch import Tensor
from numpy import mean

class Model(nn.Module):
    def __init__(self, input_shape, batch_size, loss, device="cuda:0"):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.loss = loss
        self.device = device

    def forward(self, x):
        pass

    def get_loss(self, y_pred, y_gt):

        return self.loss(y_pred, y_gt)

    def start_training(self, train_loader, val_loader, epoch=20, lr=0.01):
        self.to(self.device)

        optimizer = Adam(self.parameters(), lr=lr)

        smooth_loss = self.train_epoch(optimizer, train_loader)

        for e in range(epoch - 1):
            # train phase
            loss_train = self.train_epoch(optimizer, train_loader)
            smooth_loss = 0.99*smooth_loss + 0.01*loss_train

            # validation phase

            loss_eval = self.start_eval(val_loader)

            print('Train Epoch: {}/{}\tLoss: {:.6f}\t (Eval Loss: {:.6f})'.format(e + 1, epoch, smooth_loss, loss_eval))

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

    def start_eval(self, eval_loader):
        self.eval()
        losses = []
        for batch_idx, (x, y) in enumerate(eval_loader):
            x, y = x.to(self.device), y.float().to(self.device)
            y_pred = self.forward(x)
            loss = self.get_loss(y_pred, y)

            losses.append(loss.cpu().item())

        return mean(losses)