from models.base_model import Model
from torch import nn, sigmoid
from torch.nn.functional import relu, binary_cross_entropy


class Conv_Net(Model):
    def __init__(self, input_shape, batch_size, device="cuda:0"):

        super().__init__(input_shape, batch_size, binary_cross_entropy, device)

        self.batch_norm_1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(self.input_shape[2], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool_1 = nn.AvgPool2d(2)

        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3) # 6*6*128
        self.pool_2 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(128 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 1)

        self.act = relu
        self.final_activation = sigmoid
        

    def forward(self, x):
        x.reshape((self.batch_size, self.input_shape[2], self.input_shape[0], self.input_shape[1]))

        x = self.batch_norm_1(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))

        x = self.pool_1(x)

        x = self.batch_norm_2(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        x = self.pool_2(x)

        x = x.reshape(self.batch_size, -1)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        y = self.final_activation(self.fc3(x))

        return y
