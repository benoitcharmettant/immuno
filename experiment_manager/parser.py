from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# TODO: change name regul and regul_weight arguments to reg_type and reg_weight for consistency
def parse_args():
    parser = ArgumentParser(description='Settings for immuno therapy project.')

    parser.add_argument('--model',
                        choices=['convnet', 'convnet_1'],
                        type=str,
                        required=True,
                        help='Name of the model to be trained.')

    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Path to data root directory.')

    parser.add_argument('--logs',
                        type=str,
                        default="/",
                        required=False,
                        help='Path to log directory.')

    parser.add_argument('--protocols',
                        nargs='+',
                        help='Name of protocols to load. Eg: MK1454.',
                        required=True)

    parser.add_argument('--val_ratio',
                        type=int,
                        default=0.1,
                        required=False,
                        help='Ratio of validation images')

    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        required=False,
                        help='Training learning rate.')

    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        required=False,
                        help='Number of training epochs.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        required=False,
                        help='Training batch size.')

    parser.add_argument('--patch_size',
                        type=float,
                        default=0.40,
                        required=False,
                        help='Size of patches in centimeters.')

    parser.add_argument('--resize',
                        type=int,
                        default=40,
                        required=False,
                        help='Size of training images.')

    parser.add_argument('--regul',
                        type=float,
                        default=0,
                        required=False,
                        help='Weight of L2 regularization during training.')

    parser.add_argument('--regul_type',
                        type=str,
                        default='l2',
                        choices=['l2', 'l1'],
                        required=False,
                        help='Defines the type of regularization to apply during training. --regul must be greater '
                             'than 0 for this option to work.')

    parser.add_argument('--activation',
                        type=str,
                        default='relu',
                        choices=['relu', 'leaky_relu', 'selu', 'sigmoid'],
                        required=False,
                        help='Defines the kind of activation layer to apply')

    parser.add_argument('--seed',
                        type=str2bool,
                        default=0,
                        required=False,
                        help='Seeding option (0 / 1)')

    opt = parser.parse_args()
    return opt
