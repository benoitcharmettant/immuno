from torch import nn, sigmoid
from torch.nn.functional import relu

class Conv_Net_1(nn.Module):
    def __init__(self, input_shape, final_classes,  activation=relu, dropout=0):
        super().__init__()
        self.input_shape=input_shape
        self.dropout = dropout
        self.in_channel = input_shape[2]

        self.conv1 = nn.Conv2d(self.in_channel, 32, kernel_size=5)
        self.dropout1 = nn.Dropout2d(self.dropout)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout2 = nn.Dropout2d(self.dropout)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.pool_1 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.dropout3 = nn.Dropout2d(self.dropout)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)  # 6 * 6 * 128
        self.dropout4 = nn.Dropout2d(self.dropout)
        self.batch_norm_4 = nn.BatchNorm2d(128)
        self.pool_2 = nn.AvgPool2d(2)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=1)  # 6 * 6 * 128
        self.dropout5 = nn.Dropout2d(self.dropout)
        self.batch_norm_5 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 6 * 6, 2048)
        self.dropout6 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(2048, 512)
        self.dropout7 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(512, final_classes)

        self.act = activation
        self.final_activation = sigmoid

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)

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
