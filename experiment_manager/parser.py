from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Settings for immuno therapy project.')

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
                        help='Size of patches in centimeters')

    parser.add_argument('--resize',
                        type=int,
                        default=40,
                        required=False,
                        help='Size of training images')

    opt = parser.parse_args()
    return opt
