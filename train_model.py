from torch.utils.data import DataLoader, random_split

from models.convnet import Conv_Net
from models.data_loaders import Patch_Classifier_Dataset
from dataset.protocol import Protocol

meta_data_path = "immuno_data/recup_donnees_mk.xlsx"
prot_path = "immuno_data/MK1454"

print("Loading data...")

mk = Protocol(prot_path, meta_data_path, "MK1454")

patients = ['immuno_{}'.format(i) for i in [3, 6, 7, 10, 16]]
dataset = Patch_Classifier_Dataset([mk], [patients], 0.45, resize=40)

# TODO: ne pas définir le batch_size dès la définition du model !
model = Conv_Net((40, 40, 3), 10)

ratio_train_val = 0.9

train_size = int(ratio_train_val*len(dataset))
train_size = train_size - train_size % 10
val_size = len(dataset) - train_size

print([train_size, val_size])

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

print("Starting training...")
model.start_training(train_loader, val_loader, epoch=1000, lr=0.00001)



