from models.convnet import Conv_Net
from models.convnet_1 import Conv_Net_1
from models.squeeznet import get_squeezenet
from models.VGG import VGG
from models.ResNet import resnet_model
from torch.nn.functional import relu, leaky_relu, selu, sigmoid


def get_model(args):
    activation = get_activation_layer(args.activation)
    input_size = (args.resize, args.resize, 3)

    in_channels = 1 if args.black_white else 3

    if args.model == 'convnet':
        return Conv_Net(input_size, args.final_classes, activation=activation, dropout=args.dropout)
		
    elif args.model == 'convnet_1':
        return Conv_Net_1(input_size, args.final_classes, activation=activation, dropout=args.dropout)
		
    elif args.model.startswith('VGG'):
        return VGG(vgg_name=args.model, in_channels=in_channels, final_classes=args.final_classes, init_weights=True, batch_norm=True)

    elif args.model == 'squeezenet':
        return get_squeezenet()
		
    elif args.model.startswith('resnet'):
        return resnet_model(resnet_name=args.model, in_channels=in_channels, final_classes=args.final_classes)
		
    else:
        raise Exception('Undifined model!')


def get_activation_layer(activation):
    if activation == 'relu':
        return relu
    if activation == 'leaky_relu':
        return leaky_relu
    if activation == 'selu':
        return selu
    if activation == 'sigmoid':
        return sigmoid
