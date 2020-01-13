from models.convnet import Conv_Net
from models.convnet_1 import Conv_Net_1
from torch.nn.functional import relu, leaky_relu, selu, sigmoid


def model_manager(name, input_size, activation='relu'):

    activation = get_activation_layer(activation)

    if name == 'convnet':
        return Conv_Net(input_size, activation=activation)
    if name == 'convnet_1':
        return Conv_Net_1(input_size, activation=activation)
    return None

def get_activation_layer(activation):
    if activation == 'relu':
        return relu
    if activation == 'leaky_relu':
        return leaky_relu
    if activation == 'selu':
        return selu
    if activation == 'sigmoid':
        return sigmoid

