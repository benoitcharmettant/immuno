from torch import manual_seed
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip

from experiment_manager.logger import Logger
from experiment_manager.parser import parse_args
from models import model_manager
from dataset.data_loaders import Patch_Classifier_Dataset, get_labels_subset
from dataset.protocol import Protocol
from utils.tools import my_print
from utils.augmentation import Rotate_90

args = parse_args()
logger = Logger(args.logs, args)

data_path = args.data_path
protocols_name = args.protocols

my_print("Training args : {}".format(args), logger=logger)

if args.seed:
    my_print("Setting seed for RNG", logger=logger)
    manual_seed(101)

# Setting up model

model = model_manager(args.model, (args.resize, args.resize, 3))

# Setting up transformation

"""transformations = Compose([RandomHorizontalFlip(p=0.5),
                           RandomVerticalFlip(p=0.5),
                           RandomRotation((-180, 180), expand=False)])"""

transformations = Compose([RandomHorizontalFlip(p=0.5),
                           RandomVerticalFlip(p=0.5),
                           Rotate_90()])

# Loading dataset

my_print("   *** Loading data ***   ", logger=logger)
mk = Protocol(data_path, protocols_name[0])
# lytix = Protocol(data_path, protocols_name[1])
patients_mk = ['immuno_{}'.format(i) for i in [3, 6, 7, 10, 16]]
# patients_lytix = ['immuno_{}'.format(i) for i in [22, 24, 26, 33]] dataset = Patch_Classifier_Dataset([mk, lytix],
# [patients_mk, patients_lytix], args.patch_size, resize=args.resize, transform=transformations)

train_dataset = Patch_Classifier_Dataset([mk], [patients_mk], args.patch_size,
                                         resize=args.resize,
                                         transform=transformations,
                                         subset='train')

val_dataset = Patch_Classifier_Dataset([mk], [patients_mk], args.patch_size,
                                       resize=args.resize,
                                       transform=transformations,
                                       subset='val')

train_size = len(train_dataset)
val_size = len(val_dataset)

train_label_0, train_label_1 = get_labels_subset(train_dataset)
val_label_0, val_label_1 = get_labels_subset(val_dataset)

my_print("Number of training samples : {}, Validation samples : {}".format(train_size, val_size), logger=logger)
my_print(
    "Training labels :\t0: {}  -  1: {}  ({:.3f})".format(train_label_0, train_label_1, train_label_1 / train_size),
    logger=logger)
my_print("Validation labels :\t0: {}  -  1: {} ({:.3f})".format(val_label_0, val_label_1, val_label_1 / val_size),
         logger=logger)

p_train = train_label_1 / train_size
p_val = val_label_1 / val_size

random_pred_level = p_train * p_val + (1 - p_train) * (1 - p_val)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Training the model

my_print("*** Starting training ***", logger=logger)

model.start_training(train_loader, val_loader,
                     epoch=args.epoch,
                     lr=args.lr,
                     logger=logger,
                     regularization=args.regul,
                     random_pred_level=random_pred_level)
