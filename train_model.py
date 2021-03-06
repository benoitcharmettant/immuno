from os.path import join

from torch import manual_seed, save
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter

from experiment_manager.logger import Logger
from experiment_manager.parser import parse_args
from dataset.data_loaders import Patch_Classifier_Dataset, get_labels_subset
from dataset.protocol import Protocol
from utils.tools import my_print
from utils.augmentation import Rotate_90
from models.base_model import ModelManager

args = parse_args()
logger = Logger(args.logs, args)

data_path = args.data_path
protocols_name = args.protocols

my_print("Training args : {}".format(args), logger=logger)

if args.seed:
    my_print("Setting seed for RNG", logger=logger)
    manual_seed(101)

# Setting up model

model = ModelManager(args)

# Setting up transformation

transformations = Compose([RandomHorizontalFlip(p=0.5),
                           RandomVerticalFlip(p=0.5),
                           ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                           Rotate_90()])

# Loading dataset

my_print("   *** Loading data ***   ", logger=logger)
mk = Protocol(data_path, protocols_name[0])
# lytix = Protocol(data_path, protocols_name[1])

train_dataset = Patch_Classifier_Dataset([mk], args.patch_size,
                                         resize=args.resize,
                                         transform=transformations,
                                         subset='train',
                                         experiment=args.experiment,
                                         black_white=args.black_white)

val_dataset = Patch_Classifier_Dataset([mk], args.patch_size,
                                       resize=args.resize,
                                       transform=transformations,
                                       subset='val',
                                       experiment=args.experiment,
                                         black_white=args.black_white)

train_size = len(train_dataset)
val_size = len(val_dataset)

train_label_0, train_label_1 = get_labels_subset(train_dataset)
val_label_0, val_label_1 = get_labels_subset(val_dataset)

my_print(f"Number of training samples : {train_size}, Validation samples : {val_size}", logger=logger)
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
                     reg_weight=args.regul,
                     reg_type=args.regul_type,
                     random_pred_level=random_pred_level)

weight_path = join(logger.root_dir, "final_model.pth")
my_print(f"Saving model in {weight_path}", logger=logger)
save(model, weight_path)
