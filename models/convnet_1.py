from torch.nn import BCELoss

from models.base_model import Model
from torch import nn, sigmoid
from torch.nn.functional import relu, binary_cross_entropy


# the accuracy for the best model can be about 0.76.
class Conv_Net_1(Model):
    def __init__(self, input_shape, device="cuda:0", activation=relu, dropout=0, experiment="exp_1"):

        assert experiment in ["exp_1", 'exp_2']

        # TODO: find a way to use BCEWithLogitsLoss for better numerical stability
        if experiment == 'exp_1':
            self.final_classes = 1
            loss_function = 'bce'
        if experiment == 'exp_2':
            self.final_classes = 2
            loss_function ='bce'

        super().__init__(input_shape, loss_function, device)

        self.dropout = dropout

        self.conv1 = nn.Conv2d(self.input_shape[2], 32, kernel_size=5)
        self.dropout1 = nn.Dropout2d(self.dropout)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout2 = nn.Dropout2d(self.dropout)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.pool_1 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.dropout3 = nn.Dropout2d(self.dropout)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)  # 6*6*128
        self.dropout4 = nn.Dropout2d(self.dropout)
        self.batch_norm_4 = nn.BatchNorm2d(128)
        self.pool_2 = nn.AvgPool2d(2)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=1)  # 6*6*128
        self.dropout5 = nn.Dropout2d(self.dropout)
        self.batch_norm_5 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 6 * 6, 2048)
        self.dropout6 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(2048, 512)
        self.dropout7 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(512, self.final_classes)

        self.act = activation
        self.final_activation = sigmoid

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((-1, self.input_shape[2], self.input_shape[0], self.input_shape[1]))

        x = x.float()
        x = self.dropout1(self.act(self.batch_norm_1(self.conv1(x))))
        x = self.dropout2(self.act(self.batch_norm_2(self.conv2(x))))

        x = self.pool_1(x)

        x = self.dropout3(self.act(self.batch_norm_3(self.conv3(x))))
        x = self.dropout4(self.act(self.batch_norm_4(self.conv4(x))))
        x = self.dropout5(self.act(self.batch_norm_5(self.conv5(x))))

        x = self.pool_2(x)
        x = x.reshape(batch_size, -1)

        x = self.dropout6(self.act(self.fc1(x)))
        x = self.dropout7(self.act(self.fc2(x)))
        y = self.final_activation(self.fc3(x))
        return y
