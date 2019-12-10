import torch.nn as nn
from torch.optim import SGD, Adam


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
        # todo: gérer mieux le paramètre device, l'optimizer
        optimizer = Adam(self.parameters(), lr=lr)

        for e in range(epoch):
            self.train_epoch(optimizer, train_loader, e)

    def train_epoch(self, optimizer, train_loader, epoch):
        self.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            y_pred = self.forward(x)
            l = self.get_loss(y_pred, y)

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), l.item()))
