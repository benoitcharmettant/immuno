from os.path import join

from torch import save, tensor, norm
import torch.nn as nn
from torch.optim import Adam
from numpy import mean, append

from utils.tools import my_print, save_dictionary
from utils.metrics import accuracy_from_logits
from utils.visualisation import plot_training


class Model(nn.Module):
    def __init__(self, input_shape, loss, device="cuda:0", experiment="exp_1"):
        super(Model, self).__init__()

        self.input_shape = input_shape
        self.loss = loss
        self.device = device
        self.experiment = experiment

    def forward(self, x):
        pass

    def get_loss(self, y_pred, y_gt, reg_type='l2', reg_weight=0):
        y_pred = y_pred.reshape((-1))

        regul = reg_weight * self.get_regularization(reg_type)

        return self.loss(y_pred, y_gt) + regul

    def get_regularization(self, reg_type):
        if reg_type == 'l1':

            L1_reg = tensor(0., requires_grad=True).to(self.device)
            for name, param in self.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + norm(param, 1)

            return L1_reg
        return 0

    def train_epoch(self, optimizer, train_loader, reg_type='l2', reg_weight=0):

        self.train()

        losses = []
        metrics = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.float().to(self.device)

            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.get_loss(y_pred, y, reg_type=reg_type, reg_weight=reg_weight)

            loss.backward()
            optimizer.step()

            metric = accuracy_from_logits(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy())

            losses.append(loss.cpu().item())
            metrics.append(metric)
        return mean(losses), mean(metrics)

    def start_training(self, train_loader, val_loader, epoch=20, lr=0.01, logger=None, reg_weight=0, reg_type='l2',
                       random_pred_level=None):

        self.to(self.device)
        # TODO: add modular metrics system !
        # TODO: add options for the optimizer !
        # TODO: deal with l2 regularization the same way as it is done for l1
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=reg_weight if reg_type == 'l2' else 0)

        lowest_eval_loss = 1000

        training_results = {}
        for e in range(epoch - 1):
            # train phase
            loss_train, metric_train = self.train_epoch(optimizer, train_loader, reg_type=reg_type,
                                                        reg_weight=reg_weight)

            # validation phase

            y_pred_val, y_val, loss_val = self.evaluate(val_loader, reg_type=reg_type, reg_weight=reg_weight)
            metrics_val = accuracy_from_logits(y_pred_val, y_val)

            if loss_val < lowest_eval_loss:
                lowest_eval_loss = loss_val
                weight_path = join(logger.root_dir, "best_model.pth")
                my_print(f"Saving model in {weight_path}", logger=logger)
                save(self, weight_path)

                # save the loss and accuracy for the best model.
                if self.experiment == 'exp_1':
                    training_results['best_model_results'] = {"epoch": e + 1, "train_loss": loss_train,
                                                              "train_accuracy": metric_train,
                                                              "val_loss": loss_val, "val_accuracy": metrics_val}
                if self.experiment == 'exp_2':
                    training_results['best_model_results'] = {"epoch": e + 1, "train_loss": loss_train,
                                                              "train_accuracy": metric_train,
                                                              "val_loss": loss_val,
                                                              "val_accuracy_treatment": metrics_val[0],
                                                              "val_accuracy_injection": metrics_val[1]}

            if self.experiment == 'exp_1':
                my_print(
                    'Train Epoch: {}/{}\tLoss: {:.6f} - Acc: {:.3f}\t'
                    '(Eval Loss: {:.6f} - Acc: {:.3f})'.format(e + 1,
                                                               epoch,
                                                               loss_train,
                                                               metric_train,
                                                               loss_val,
                                                               metrics_val),
                    logger=logger)

            if self.experiment == 'exp_2':
                my_print(
                    'Train Epoch: {}/{}\tLoss: {:.6f} - Acc: {:.3f}\t'
                    '(Eval Loss: {:.6f} - Acc: {:.3f} - Acc_bis: {:.3f})'.format(e + 1,
                                                                                 epoch,
                                                                                 loss_train,
                                                                                 metric_train,
                                                                                 loss_val,
                                                                                 metrics_val[0],
                                                                                 metrics_val[1]),
                    logger=logger)

            if e > 0 and e % 100 == 0 and logger is not None:
                plot_training(logger.root_dir, random_pred_level=random_pred_level)

        # save the loss and accuracy for the final model.

        if self.experiment == 'exp_1':
            training_results['final_model_results'] = {"epoch": e + 1, "train_loss": loss_train,
                                                       "train_accuracy": metric_train,
                                                       "val_loss": loss_val, "val_accuracy": metrics_val}

        if self.experiment == 'exp_2':
            training_results['final_model_results'] = {"epoch": e + 1, "train_loss": loss_train,
                                                       "train_accuracy": metric_train,
                                                       "val_loss": loss_val,
                                                       "val_accuracy_treatment": metrics_val[0],
                                                       "val_accuracy_injection": metrics_val[1]}

        # save the results in a txt file.
        training_results_file = join(logger.root_dir, "training_results.txt")
        save_dictionary(training_results, training_results_file)

    def evaluate(self, val_loader, reg_type='l2', reg_weight=0):
        self.eval()

        gt = None
        preds = None
        losses = []

        for (x, y) in val_loader:

            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.forward(x)

            loss = self.get_loss(y_pred.float(), y.float(), reg_type=reg_type, reg_weight=reg_weight)
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
