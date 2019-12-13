from os.path import join

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

from experiment_manager.logger import Logger
from experiment_manager.parser import parse_args
from models.convnet import Conv_Net
from models.data_loaders import Patch_Classifier_Dataset
from dataset.protocol import Protocol
from utils.tools import my_print

args = parse_args()
logger = Logger(args.logs)

data_path = args.data_path
protocols_name = args.protocols

my_print("Training args : {}".format(args), logger=logger)

# Setting up model

model = Conv_Net((args.resize, args.resize, 3))


# Setting up transformation

transformations = Compose([RandomHorizontalFlip(p=0.5),
                          RandomVerticalFlip(p=0.5),
                          RandomRotation((-180, 180), expand=False)])

transformations = Compose([RandomHorizontalFlip(p=0.5),
                          RandomVerticalFlip(p=0.5)])

# Loading dataset

my_print("   *** Loading data ***   ", logger=logger)
mk = Protocol(data_path, protocols_name[0])
patients = ['immuno_{}'.format(i) for i in [3, 6, 7, 10, 16]]
dataset = Patch_Classifier_Dataset([mk], [patients], args.patch_size, resize=args.resize, transform=transformations)


# Splitting between training and validation set

ratio_train_val = args.val_ratio
train_size = int((1 - ratio_train_val)*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

my_print("Number of training samples : {}, Validation samples : {}".format(train_size, val_size), logger=logger)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Training the model

my_print("*** Starting training ***", logger=logger)

model.start_training(train_loader, val_loader,
                     epoch=args.epoch,
                     lr=args.lr,
                     logger=logger,
                     regularization=args.regul)



