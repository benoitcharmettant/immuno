from cv2 import resize
from numpy import array, transpose
from PIL import Image

from torch.utils.data import Dataset

from utils.image import get_patches, get_ls_patch_coord, get_dict_split


class Patch_Classifier_Dataset(Dataset):
    def __init__(self, ls_protocols, allowed_patients_by_protocol, patch_size, resize,
                 transform=None,
                 subset='train'):  # patch size in centimeter

        assert len(ls_protocols) == len(allowed_patients_by_protocol)
        assert subset in ['train', 'val']

        self.ls_protocols = ls_protocols
        self.patch_size = patch_size
        self.resize = resize
        self.transform = transform

        self.patches = []
        self.labels = []
        self.coord = []

        for i in range(len(self.ls_protocols)):
            self.collect_data(self.ls_protocols[i], allowed_patients_by_protocol[i])

        self.patches = array(self.patches)
        self.patches = self.patches.reshape((-1, 3, 40, 40))

    def __getitem__(self, index):

        patch = self.patches[index]

        if self.transform:
            patch = transpose(patch, (1, 2, 0)) * 255
            patch = Image.fromarray(patch.astype('uint8'))
            patch_tf = self.transform(patch)

            patch = transpose(array(patch_tf), (2, 0, 1)) / 255

        return patch, self.labels[index]

    def __len__(self):
        return len(self.patches)

    def new_patch(self, patch, label, coord):
        patch = resize(patch, (self.resize, self.resize))
        patch = patch[:, :, :3]
        self.patches.append(patch)
        self.labels.append(label)
        self.coord.append(coord)

    def collect_data(self, protocol, allowed_patients):
        for patient in protocol.ls_patients:
            if patient.name in allowed_patients:

                debut_tt = patient.get_patient_info()['debut_tt']

                for tumor in patient.ls_tumors.values():
                    for exam in tumor.get_exams().values():
                        label_exam = 0
                        if (exam[0]['date'] - debut_tt).days > 0:
                            label_exam = 1
                        for image in exam:
                            ls_patches = get_patches(image, protocol.root_data_path, self.patch_size)
                            ls_coord = get_ls_patch_coord(image, protocol.root_data_path)
                            dict_split = get_dict_split(image, protocol.root_data_path)

                            for i, patch in enumerate(ls_patches):
                                if dict_split[f'{ls_coord[i][0]}_{ls_coord[i][0]}'] == self.subset:
                                    self.new_patch(patch, label_exam, ls_coord[i])


def get_labels_subset(dataset):
    nb_label_1 = sum(dataset.labels)
    return len(dataset) - nb_label_1, nb_label_1
