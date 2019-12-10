from torch.utils.data import DataLoader

from models.convnet import Conv_Net
from models.data_loaders import Patch_Classifier_Dataset
from dataset.protocol import Protocol

meta_data_path = "C:/Users/b_charmettant/Desktop/Données_immunothérapies/MK1454/recup_donnees_mk.xlsx"
prot_path = "C:/Users/b_charmettant/data/immuno/MK1454"
mk = Protocol(prot_path, meta_data_path, "MK1454")
print("Loading data...")
patients = ['immuno_{}'.format(i) for i in [3, 6, 7, 10, 16]]
dataset = Patch_Classifier_Dataset([mk], [patients], 0.40, resize=40)

model = Conv_Net((40, 40, 3), 1)

train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

print("Starting training...")
model.start_training(train_loader, None)



