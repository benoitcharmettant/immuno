from models.convnet import Conv_Net
from models.convnet_1 import Conv_Net_1
from torch.nn.functional import relu, leaky_relu, selu, sigmoid



def get_model(args):
    activation = get_activation_layer(args.activation)
    input_size = (args.resize, args.resize, 3)
    if args.model == 'convnet':
        return Conv_Net(input_size, args.final_classes, activation=activation, dropout=args.dropout)
    if args.model == 'convnet_1':
        return Conv_Net_1(input_size, args.final_classes, activation=activation, dropout=args.dropout)
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
